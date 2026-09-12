"""Sweep the grouped-MMQ tile geometry on real 256-expert GGUF banks.

A synthetic 32-expert microbenchmark understates the one cost that grows with ``mmq_x``:
``moe_align_block_size`` pads every expert's routed-row list up to a whole ``mmq_x``, so
with 256 experts and 2048x8 routed rows a wide column tile can double the rows the kernel
actually multiplies. The win it buys is weight traffic -- a weight tile is re-read
``ceil(rows_of_this_expert / mmq_x)`` times -- so the optimum is a real trade-off and has
to be measured here, not copied from llama.cpp (which has no grouped variant).

Real checkpoint bytes, so the same run also reports MMQ-vs-MMVQ agreement: random uint8
is not a valid K-quant super-block and would make the correctness column meaningless.

Run once per geometry, with a distinct extensions dir so each one really recompiles::

    for g in 4/32/4 8/64/4 16/64/4 16/128/4 32/64/4 32/128/4 64/64/4 64/128/4; do
      FREETOKEN_GGUF_MOE_TILES="Q4_K:$g Q5_K:$g" \\
      TORCH_EXTENSIONS_DIR=/tmp/ftx_$(echo $g | tr / _) \\
      python benchmarks/gguf_mmq_tilesweep.py --model model.gguf --label "$g"
    done

Result on an RTX 3060 (ms per MoE layer at a 2048-token chunk)::

    4/32/4 (as shipped) 42.94    16/128/4  12.42    64/128/4  11.52
    8/64/4              18.11    32/128/4  10.75    64/64/4   12.41
    16/64/4             13.02    32/64/4   11.11

-- hence the compiled-in MoE default of 32/128/4.
"""

import argparse
import os
import time

import torch

from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_a8_vec, ggml_moe_get_block_size
from freetoken.kernel.triton.moe_align import (
    moe_align_block_size,
    sanitize_moe_align_tail,
)
from freetoken.models.gguf.reader import iter_gguf_tensors

GGML_Q4_K, GGML_Q5_K = 12, 13
DT = torch.bfloat16


def align(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    flat = topk_ids.reshape(-1, 1).contiguous().to(torch.int32)
    sorted_ids, expert_ids, npp = moe_align_block_size(flat, block_size, num_experts)
    sorted_ids, expert_ids = sanitize_moe_align_tail(
        sorted_ids, expert_ids, npp, block_size, flat.numel()
    )
    return sorted_ids, expert_ids, npp


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def run_one(tensors, name, quant_type, k_in, x_rows, top_k, num_experts, device):
    t = tensors[name]
    per_expert = t.rows // num_experts * t.row_bytes
    weight = t.packed().reshape(num_experts, per_expert).contiguous().unsqueeze(-1).to(device)
    rows = t.rows // num_experts

    x = torch.randn(x_rows, k_in, dtype=DT, device=device) * 0.5
    ids = torch.randint(0, num_experts, (x_rows, top_k), dtype=torch.int32, device=device)
    block = ggml_moe_get_block_size(quant_type)
    sorted_ids, expert_ids, npp = align(ids, block, num_experts)

    ms = timeit(
        lambda: ggml_moe_a8(
            x, weight, sorted_ids, expert_ids, npp, quant_type, rows, top_k, x_rows
        )
    )
    ref = ggml_moe_a8_vec(x, weight, ids, top_k, quant_type, rows, x_rows).float()
    got = ggml_moe_a8(
        x, weight, sorted_ids, expert_ids, npp, quant_type, rows, top_k, x_rows
    ).float()
    rel = ((ref - got).abs().max() / ref.abs().max()).item()
    pad = int(npp.item()) / (x_rows * top_k)

    del weight, x, ref, got
    torch.cuda.empty_cache()
    return ms, rel, block, pad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("FT_GGUF_MODEL"), required=False)
    ap.add_argument("--label", default="?")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--moe-intermediate", type=int, default=512)
    ap.add_argument("--layer", default="blk.0")
    args = ap.parse_args()
    if not args.model:
        ap.error("--model (or FT_GGUF_MODEL) is required")

    device = torch.device("cuda")
    gate_name = f"{args.layer}.ffn_gate_exps.weight"
    down_name = f"{args.layer}.ffn_down_exps.weight"
    want = {gate_name, down_name}
    tensors = {t.name: t for t in iter_gguf_tensors(args.model) if t.name in want}

    g_ms, g_rel, g_blk, g_pad = run_one(
        tensors, gate_name, GGML_Q4_K, args.hidden, args.tokens, args.topk,
        args.experts, device,
    )
    d_ms, d_rel, d_blk, d_pad = run_one(
        tensors, down_name, GGML_Q5_K, args.moe_intermediate,
        args.tokens * args.topk, 1, args.experts, device,
    )
    # The production gate_up bank is gate|up fused: same K, twice the rows, twice the time.
    layer_ms = 2 * g_ms + d_ms
    print(
        "SWEEP %-14s blk=%2d/%2d  gate(x2) %7.3f ms  down %7.3f ms  layer %7.3f ms  "
        "pad %.2fx/%.2fx  rel %.4f/%.4f"
        % (args.label, g_blk, d_blk, 2 * g_ms, d_ms, layer_ms, g_pad, d_pad, g_rel, d_rel)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
