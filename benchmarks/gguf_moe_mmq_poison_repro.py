"""Deterministic reproducer for the out-of-bounds access in the ggml MoE MMQ kernel.

``moe_align_block_size`` allocates ``sorted_token_ids`` and ``expert_ids`` with
``torch.empty`` and only writes the region up to ``num_tokens_post_padded``.
``ggml_moe_a8``'s grid, however, covers ``floor(len(sorted_token_ids) / mmq_x)`` column
blocks -- including blocks past ``npp``, whose ``expert_ids`` and ``sorted_token_ids``
entries are therefore whatever the caching allocator last left there.

``csrc/gguf/moe.cuh`` guards with ``exp_idx > 255 || exp_idx < 0`` and
``col_dst >= ncols_dst``. Neither catches a NEGATIVE token offset: ``col_y_eff <
ncols_y`` is true for a negative (the activations are read out of bounds) and
``col_dst >= ncols_dst`` is false for a negative (``dst`` is WRITTEN out of bounds). In
a short-lived process the tail happens to be zeros; in a long-running server it is
recycled memory, and the kernel eventually faults.

Usage::

    python benchmarks/gguf_moe_mmq_poison_repro.py <clean|poison|sanitized> \\
        --model /path/to/model.gguf

Expected: ``clean`` and ``sanitized`` finish with finite output; ``poison`` dies with
``CUDA error: an illegal memory access was encountered``.
"""

import argparse
import os
import sys

import torch

from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
from freetoken.kernel.triton.moe_align import (
    moe_align_block_size,
    sanitize_moe_align_tail,
)
from freetoken.models.gguf.reader import iter_gguf_tensors

GGML_Q4_K = 12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("clean", "poison", "sanitized"))
    ap.add_argument(
        "--model",
        default=os.environ.get("FT_GGUF_MODEL"),
        help="path to a Qwen3.5/3.6-class MoE GGUF (or set FT_GGUF_MODEL)",
    )
    ap.add_argument("--tensor", default="blk.0.ffn_gate_exps.weight")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2048, help="model hidden size (K)")
    ap.add_argument(
        "--n2", type=int, default=1024, help="fused gate|up output width (2 * moe_inter)"
    )
    args = ap.parse_args()

    if not args.model:
        ap.error("--model (or FT_GGUF_MODEL) is required")
    if not torch.cuda.is_available():
        print("needs a CUDA device", file=sys.stderr)
        return 2

    dev = torch.device("cuda")
    E, TOPK, T = args.experts, args.topk, args.tokens

    tensor = next(t for t in iter_gguf_tensors(args.model) if t.name == args.tensor)
    gate = tensor.packed().reshape(E, -1)
    # gate|up fused, as the engine stores it.
    gate_up = torch.cat([gate, gate], dim=1).contiguous().unsqueeze(-1).to(dev)

    x = torch.randn(T, args.hidden, dtype=torch.bfloat16, device=dev) * 0.3
    ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=dev)

    block = ggml_moe_get_block_size(GGML_Q4_K)
    flat = ids.reshape(-1, 1).contiguous()
    sorted_ids, expert_ids, npp = moe_align_block_size(flat, block, E)
    used = int(npp.item())
    print(
        "mode=%s sorted_len=%d npp=%d expert_ids=%d grid_y=%d used_blocks=%d"
        % (args.mode, sorted_ids.numel(), used, expert_ids.numel(),
           sorted_ids.numel() // block, used // block),
        flush=True,
    )

    if args.mode in ("poison", "sanitized"):
        # What a recycled allocator page looks like: negative junk past npp, and a
        # perfectly VALID expert id, so the exp_idx guard passes.
        sorted_ids[used:] = -999_999
        expert_ids[used // block:] = 7
    if args.mode == "sanitized":
        sorted_ids, expert_ids = sanitize_moe_align_tail(
            sorted_ids, expert_ids, npp, block, flat.numel()
        )

    out = ggml_moe_a8(
        x, gate_up, sorted_ids, expert_ids, npp, GGML_Q4_K, args.n2, TOPK, T
    )
    torch.cuda.synchronize()
    print(
        "OK: no fault, out=%s finite=%s"
        % (tuple(out.shape), bool(torch.isfinite(out).all())),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
