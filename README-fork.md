# FreeToken fork: native GGUF for Qwen3.5/3.6-class MoE

A fork of [FreeToken](https://github.com/FlashML-org/FreeToken) `v0.1.2` that runs a
Qwen3.5/3.6-class MoE **GGUF** checkpoint natively — the expert banks and most of the
dense tensors stay in their packed ggml blocks from disk to kernel, and are never
materialized as bf16.

Upstream's own `README.md` is unchanged; this file only describes what the fork adds.

---

## What is in here

The branch is ordered so the bug fixes come first and stand alone. **The first four
commits are useful to anyone using this engine, GGUF or not**, and three of them describe
code that also exists in sgl-kernel and vLLM:

| | commit | |
|---|---|---|
| 1 | `fix(gguf)` | `moe_align_block_size` leaves its buffer tails at `torch.empty`, and `ggml_moe_a8` launches column blocks past `num_tokens_post_padded`. `moe.cuh`'s guards do not reject a **negative** token offset, so the kernel reads *and writes* out of bounds. Benign on fresh memory, fatal in a long-running server. Reproducer included. |
| 2 | `fix(gguf)` | The MoE MMQ kernel fills its q8_1 scale tile for `nwarps` columns, not `mmq_x`. Latent — correct only because the shipped tile has `mmq_x == nwarps == 4`. Widening the tile without this produces NaN. |
| 3 | `fix(gguf)` | The vLLM port of llama.cpp's MMQ kept the real per-architecture tile sizes behind `#if defined(USE_ROCM)`, so **every CUDA build** got `MMQ_X_Q4_K 4` instead of 64 and re-read each weight tile `ncols_y/mmq_x` times. Measured 4.01x on the grouped MoE path and 6–7x on dense (`qkv_proj` 2.28 → 16.0 TOP/s). |
| 4 | `fix(moe)` | Hybrid decode marks CPU-assigned routes `-1`, zeroes their router weight, then passes `topk_ids.clamp_min(0)` to the GEMV anyway — so the GPU decodes them through slot 0. With mixed-type banks that parses another layer's bytes under the wrong block layout, and `0 * NaN` is NaN. |
| 5–9 | `feat` | The GGUF work proper: K-quant type metadata, the `qwen35moe` adapter and its packed-bank MoE kernels, packed dense weights, an expert-bank stride taken from the tensor, and K-quant CPU dot products that make `--moe-backend hybrid` reachable. |
| 10 | `fix(gguf)` | The GGUF tokenizer registered 3 of 27 CONTROL tokens and no USER_DEFINED ones, so `<think>` tokenized as `['<th', 'ink', '>']` — silently degrading **every** Qwen3.5/3.6 chat prompt. |
| 11–12 | `feat` | A sweepable hybrid fetch fraction, and opt-in decode-step profiling. |

---

## Measured

One box, one day, same engine, same server harness:

| checkpoint | decode tok/s | prefill tok/s |
|---|---:|---:|
| `Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_M.gguf` (this fork) | **53.9** | **1615** |
| Qwen3.6-35B-A3B **NVFP4** safetensors (upstream path) | **57.1** | 960 |

Both at context 131072, `--moe-backend hybrid`, fetch fraction 0.28, tuned cache sizes
(2850 slots for the GGUF, 3000 for NVFP4).

**Read that honestly: it is −6% decode and +68% prefill, not a clean win.** If your
workload is dominated by generation on a checkpoint you already have in NVFP4, upstream
is still faster. What this fork buys you is the ability to run the GGUF you actually
have — the format most community fine-tunes and Dynamic quants ship in — at roughly
NVFP4 decode speed and materially better prefill, instead of not at all.

For scale, the owner's `llama.cpp` on a comparable Q4_K_M with `--n-cpu-moe 20..23` does
~52–55 tok/s decode on the same box.

### Hardware

* GPU: **NVIDIA RTX 3060, 12 GB** (11.63 GiB usable), SM 8.6
* CPU: **Ryzen 5 5500** (6C/12T, AVX2, no AVX-512)
* RAM: **31 GiB**
* Weights on NVMe

That is a deliberately small box. The whole point of the offload/hybrid path is a 35B-A3B
model that does not fit, and every number above is a 12 GB-GPU number.

---

## What is verified, and what is not

**Verified** (on the hardware above, on two Qwen3.6-class Q4_K_M checkpoints):

* end-to-end generation quality against the same architecture's NVFP4 checkpoint;
* MMQ output bit-comparable against MMVQ on real packed checkpoint bytes;
* the K-quant CPU dot products against the GPU path and against reference dequantization;
* GGUF special-token round-tripping against llama.cpp's tokenization;
* stability: 56-request mixed stress (tiny prompts, ~4600-token prompts crossing the
  chunk boundary, prefix-cache hits) with zero crashes, twice, on the configuration that
  used to die at request 15.
* **this branch's own source, executed.** The measurements above were taken with the
  patch applied in place; this branch additionally lifts `sanitize_moe_align_tail()` out
  of `_mmq_align` into `kernel/triton/moe_align.py`, so commit 1 stands alone. That
  refactor was then installed into the venv and run: Cyber-Tiel-Coder-35B-A3B Q4_K_M at
  ctx 131072 measured **decode 52.7 tok/s (best 53.4) / prefill 1585 tok/s**, against
  53.3 (53.9) / 1588 for the in-place patch — run-to-run noise — with smoke 5/5,
  stress 56/56 and a 66.9k-token prompt served. The branch is not just diff-equivalent
  to what was benchmarked; it is what ran.

**Not verified:**

* **any GPU other than an RTX 3060.** The MMQ tile geometry is llama.cpp's ≥ Volta
  numbers for the dense path and a locally swept 32/128/4 for the MoE path. Both are
  overridable per type at runtime (`FREETOKEN_GGUF_MMQ_TILES`, `FREETOKEN_GGUF_MOE_TILES`)
  precisely because the MoE optimum is a real trade-off between weight reuse and
  routed-row padding, and it will move on other hardware. Re-sweep with
  `benchmarks/gguf_mmq_tilesweep.py`.
* **ROCm.** The tile-size fix rewrites the CUDA branch and leaves the `USE_ROCM` branch
  alone, so it should be inert there, but nothing here was built or run on ROCm.
* **AVX-512.** The CPU K-quant kernels use ggml's AVX2 branches with a scalar fallback;
  ggml's AVX-512/NEON variants were not ported.
* **Multi-GPU / tensor parallel.** Single-GPU only.
* **Any architecture other than `qwen35moe`.** The adapter is model-specific; the kernel
  and tokenizer fixes are not.
* **`_prefill_routed`'s chunk cost** — see limitations.

---

## Known limitations

* **No IQ-quant support.** Only Q2_K–Q6_K, Q4_0 and Q8_0 are handled. IQ quants
  (`IQ4_XS`, `IQ2_S`, …) have CUDA kernels in the vendored sources but no Python-side
  metadata, no CPU dot products, and were never tested.
* **Host memory.** A 35B-A3B Q4_K_M keeps roughly **19 GiB of pinned host banks**. On
  this 31 GiB box that works, but it is tight, and the `rba/FreeToken-ROCm` fork's README
  explicitly warns that this configuration is unreliable on 32 GB machines. Treat 32 GB
  as the floor, not the target, and watch for orphaned scheduler children holding pinned
  memory across restarts — they will push the box into swap and every measurement with
  it.
* **Prefill streams whole expert layers.** `_prefill_routed` fetches all 256 experts of
  all 40 layers for any prefill chunk — 19.4 GiB per chunk, served partly device-side and
  partly over PCIe — because a full 2048-token chunk really does route to essentially
  every expert (2048 × top-8 over 256 experts hits 255.7 of them in expectation). The
  cost is **per chunk, not per token**, so a two-token continuation on a fully cached
  3200-token prefix pays the same ~1.1 s as a full chunk. That is unaddressed here. A
  small-chunk path modelled on `deepseek_v4`'s is the obvious fix and is not in this
  branch.
* **The CPU MoE extension ships prebuilt with no JIT path**, so its `.so` must be rebuilt
  for the K-quant kernels to reach a running server; `--moe-backend cpu/hybrid` fails
  loudly rather than computing wrong experts if it is stale.
* **Bring-up instrumentation is still present** behind env vars
  (`FREETOKEN_GGUF_K_DEBUG`, `FREETOKEN_GGUF_K_MMQ_TRACE`, `FT_DECODE_PROF`, …). All of
  it is off by default and costs at most one `is None` test.

---

## Attribution

* **llama.cpp / ggml (MIT).** The CPU K-quant dot products
  (`ggml_vec_dot_q4_K_q8_K` / `_q5_K_` / `_q6_K_` and their scalar twins),
  `quantize_row_q8_K_ref`, `nearest_int`, the scale-shuffle byte tables and the
  `block_q4_K` / `block_q5_K` / `block_q6_K` / `block_q8_K` layouts are **ported from
  llama.cpp**, not written here. Per-function `file:line` provenance is in the source
  header of `kernel/csrc/cpu_moe/cpu_moe_ext.cpp`, marked `FT-CPU-KQUANT`. The MMQ tile
  numbers restored in commit 3 are likewise llama.cpp's own (b2899, `ggml-cuda/mmq.cu`).
  Thanks to the ggml authors. Apache-2.0 and MIT are compatible; the notice is not
  optional.
* **The vendored `csrc/gguf/*` kernels** came into FreeToken from vLLM's copy of
  llama.cpp b2899; the bugs fixed in commits 1–3 are in that lineage, not introduced by
  this fork.
* **`rba/FreeToken-ROCm`, issue #122.** That fork arrived independently at the same
  packed-expert-bank design. This fork's expert-bank stride fix (commit 8) reaches the
  same per-expert slot size as their proposed `FT_GGUF_BANK_PROMOTE`, by a different
  route — see the commit message for why promote-up was rejected here. Their README's
  warning about host-memory pressure on 32 GB boxes is repeated above because it is
  correct.
* **FreeToken** itself is Apache-2.0; this fork is too.
