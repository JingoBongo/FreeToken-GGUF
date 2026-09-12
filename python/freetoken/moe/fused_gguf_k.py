"""Grouped expert GEMV/GEMM over native GGUF K-quant banks (borrowed ggml MoE kernels).

Generalizes :mod:`freetoken.moe.fused_q4_0` in two ways, both forced by real GGUFs:

* the block type may be any ggml type the vendored kernels dispatch (Q2_K..Q6_K, Q4_0,
  Q8_0), not just Q4_0;
* the fused ``gate_up`` bank and the ``down`` bank may carry *different* types, and the
  type varies per layer -- Unsloth-Dynamic quants deliberately spend more bits on some
  tensors and some layers (a UD-Q4_K_M has Q4_K gate/up throughout but Q5_K down on 37
  layers and Q6_K on three).

The banks are padded to the widest row over layers so the offload cache can keep one
shape for every layer. ``ggml_moe_a8_vec`` (MMVQ) derives the expert stride entirely
from the quant type and the padded ``nrows`` the caller passes -- ``x = vx + expert *
nrows * blocks_per_row`` (moe_vec.cuh) -- so each expert's rows are written contiguously
from the start of its region, the kernel reads exactly its own bytes, and the slack sits
unread in the tail (sliced off the output).

Prefill uses ``ggml_moe_a8`` (MMQ) instead. MMVQ is a GEMV: one output row per thread
block, the weight re-read once per routed token, which is right for a decode step's 8
routed rows and ~3x off the pace at prefill batch sizes (measured on this checkpoint's
geometry, RTX 3060, 2048-token chunk: gate_up 72.3 ms MMVQ vs 24.5 ms MMQ; down 71.5 ms
vs 15.8 ms). MMQ also takes the expert stride from ``W.stride(0)`` rather than deriving
it from ``nrows``, so it reads the padded bank with the REAL row count and the padding
costs nothing on that path (verified bit-comparable against MMVQ on real packed bytes).
"""

from __future__ import annotations

import os

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}

# Temporary bring-up instrumentation: FREETOKEN_GGUF_K_DEBUG=N prints the first N calls.
_DEBUG: dict | None = None

# Measurement knob: FREETOKEN_GGUF_K_PREFILL=mmvq forces the old GEMV path at prefill so
# an A/B isolates the MMQ win. Anything else (default) uses MMQ above _MMQ_MIN_TOKENS.
_PREFILL_KERNEL = os.getenv("FREETOKEN_GGUF_K_PREFILL", "mmq").lower()
# FREETOKEN_GGUF_K_MMQ_TRACE=1 logs each MMQ call's host-side shapes (no device sync, so
# it does not perturb the timing the way the .item() debug dump does).
_MMQ_TRACE = os.getenv("FREETOKEN_GGUF_K_MMQ_TRACE", "") in ("1", "true", "yes", "on")
# FREETOKEN_GGUF_K_ALIGN_ONLY=1 runs MMQ's alignment but keeps the MMVQ GEMV -- an
# isolation probe for the fault documented in PERF.md §5.
_ALIGN_ONLY = os.getenv("FREETOKEN_GGUF_K_ALIGN_ONLY", "") in ("1", "true", "yes", "on")
# Below this the per-call moe_align + q8_1 setup outweighs MMQ's tiling win.
_MMQ_MIN_TOKENS = 8


def _debug_state():
    """Debug state, or None when printing would be unsafe.

    The stats below call .item(), which synchronizes with the host -- illegal while a
    CUDA graph is capturing (cudaErrorStreamCaptureUnsupported), so skip capture.
    """
    global _DEBUG
    if _DEBUG is None:
        n = int(os.getenv("FREETOKEN_GGUF_K_DEBUG", "0") or 0)
        _DEBUG = {"limit": n, "n": 0}
    if not _DEBUG["limit"]:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    return _DEBUG


def _mmq_align(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """``moe_align_block_size`` over the FLAT routed-row list, with the unwritten tail
    made inert.

    gate_up and down route identically (row i of the flattened ``topk_ids`` picks the
    same expert for both), and MMQ reads its x row as ``sorted_token_ids[i] / top_k`` --
    so one alignment serves the gate_up call (top_k = the real top_k, x has num_tokens
    rows) and the down call (top_k = 1, x has num_tokens*top_k rows) unchanged.

    The sanitize is load-bearing, not defensive: ``ggml_moe_a8`` reads and writes past
    ``num_tokens_post_padded``, where ``moe_align_block_size`` leaves whatever
    ``torch.empty`` picked up. See ``sanitize_moe_align_tail`` for the mechanism and
    ``benchmarks/gguf_moe_mmq_poison_repro.py`` for the reproducer.
    """
    from freetoken.kernel.triton.moe_align import (
        moe_align_block_size,
        sanitize_moe_align_tail,
    )

    ids = topk_ids.reshape(-1, 1)
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    ids = ids.contiguous()
    sorted_ids, expert_ids, num_post = moe_align_block_size(ids, block_size, num_experts)
    # sentinel == ids.numel() == ncols_dst for both calls; the epilogue drops it.
    sorted_ids, expert_ids = sanitize_moe_align_tail(
        sorted_ids, expert_ids, num_post, block_size, ids.numel()
    )
    return sorted_ids, expert_ids, num_post


def fused_experts_gguf_k(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, gate_up_bytes, 1] uint8
    down_q: torch.Tensor,  # [num_slots, down_bytes, 1] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    gate_up_type: int,
    down_type: int,
    gate_up_rows: int,
    down_rows: int,
    n2: int,
    h: int,
    is_prefill: bool = False,
) -> torch.Tensor:
    """``*_rows`` is the padded row count this layer's quant implies for its bank (see
    ``qwen3_5_moe.gguf._bank_geometry``); ``n2``/``h`` are the real output widths, which
    the padded tail is sliced back down to.

    ``is_prefill`` picks MMQ over MMVQ (see the module docstring); MMQ addresses experts
    by ``W.stride(0)``, so it is handed ``n2``/``h`` -- the real widths -- and never
    touches the padding."""
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_a8_vec, ggml_moe_get_block_size

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]

    _dbg = _debug_state()
    if _dbg is not None and _dbg["n"] < _dbg["limit"]:
        _dbg["n"] += 1
        x = hidden_states.float()
        print(
            "[gguf_k#%d] in: shape=%s finite=%s absmax=%.4g | gate_up %s type=%d | "
            "down %s type=%d | topk_ids max=%d dtype=%s"
            % (_dbg["n"], tuple(hidden_states.shape), bool(x.isfinite().all()),
               x.abs().max().item(), tuple(gate_up_q.shape), int(gate_up_type),
               tuple(down_q.shape), int(down_type), int(topk_ids.max().item()),
               topk_ids.dtype),
            flush=True,
        )

    use_mmq = is_prefill and _PREFILL_KERNEL == "mmq" and num_tokens >= _MMQ_MIN_TOKENS
    if _ALIGN_ONLY and is_prefill and not use_mmq:
        # Isolation probe: run MMQ's moe_align_block_size and throw the result away, so a
        # stress run tells apart "the triton alignment faults" from "ggml_moe_a8 faults".
        _mmq_align(topk_ids, ggml_moe_get_block_size(int(gate_up_type)), gate_up_q.shape[0])
    if use_mmq:
        # Prefill: position == expert id, so the bank's row count IS the expert count.
        num_experts = gate_up_q.shape[0]
        blk = ggml_moe_get_block_size(int(gate_up_type))
        blk_dn = ggml_moe_get_block_size(int(down_type))
        sorted_ids, expert_ids, num_post = _mmq_align(topk_ids, blk, num_experts)
        if _MMQ_TRACE:
            # Host-side values only: printing these must not synchronize, or it would
            # serialize the very overlap we are trying to observe.
            print("[mmq] T=%d top_k=%d E=%d n2=%d h=%d gu_t=%d dn_t=%d blk=%d/%d "
                  "sorted=%d eids=%d x=%s%s inter_later" %
                  (num_tokens, top_k, num_experts, n2, h, int(gate_up_type), int(down_type),
                   blk, blk_dn, sorted_ids.numel(), expert_ids.numel(),
                   tuple(hidden_states.shape),
                   "" if hidden_states.is_contiguous() else " NONCONTIG"), flush=True)
        # gate_up: [num_tokens*top_k, n2] straight out -- no padded slice needed.
        gate_up = ggml_moe_a8(
            hidden_states, gate_up_q, sorted_ids, expert_ids, num_post,
            int(gate_up_type), n2, top_k, num_tokens,
        )
        inter = act_fn(gate_up)
        if blk_dn != blk:
            sorted_ids, expert_ids, num_post = _mmq_align(topk_ids, blk_dn, num_experts)
        out = ggml_moe_a8(
            inter, down_q, sorted_ids, expert_ids, num_post,
            int(down_type), h, 1, num_tokens * top_k,
        )
    else:
        # gate_up: [num_tokens*top_k, gate_up_rows] -> slice to the real 2I -> activation
        gate_up = ggml_moe_a8_vec(
            hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_type), gate_up_rows,
            num_tokens,
        )
        if gate_up_rows != n2:
            gate_up = gate_up[:, :n2].contiguous()
        inter = act_fn(gate_up)
        # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
        out = ggml_moe_a8_vec(
            inter, down_q, topk_ids, 1, int(down_type), down_rows, num_tokens * top_k
        )
        if down_rows != h:
            out = out[:, :h].contiguous()

    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    out = out.sum(dim=1)
    if _dbg is not None and _dbg["n"] <= _dbg["limit"]:
        gu, it, ot = gate_up.float(), inter.float(), out.float()
        print(
            "[gguf_k#%d] mmq=%s gate_up finite=%s absmax=%.4g | act finite=%s absmax=%.4g | "
            "out finite=%s absmax=%.4g"
            % (_dbg["n"], use_mmq, bool(gu.isfinite().all()), gu.abs().max().item(),
               bool(it.isfinite().all()), it.abs().max().item(),
               bool(ot.isfinite().all()), ot.abs().max().item()),
            flush=True,
        )
    return out


__all__ = ["fused_experts_gguf_k"]
