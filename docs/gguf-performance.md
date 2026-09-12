# GGUF performance: what moved it, what did not, and why

Companion to [README-fork.md](../README-fork.md). That file says what the fork is; this
one is the optimization record — **including every lever that paid nothing**, which is
the half that saves someone else the week.

The naive adapter ran at **27.4 tok/s decode / 315 tok/s prefill**. It now does
**53.9 / 1615** at ctx 131072 on an RTX 3060 12 GB. The first explanation anyone reaches
for — "K-quants dequantize in-kernel, GGUF is just structurally slower" — was wrong, and
believing it would have stopped the work at 27.4.

> **Two eras of numbers below.** The lever-by-lever progression was measured at
> **ctx 32768** (ending at 57.7 tok/s), then everything was re-based to the real working
> window **ctx 131072** (53.9). The context costs 3.5 tok/s, not a collapse — see
> [Context](#context-is-cheaper-than-it-looks). Each table says which era it is from.

---

## 1. Where the decode time actually went

Do this first. Every wrong guess in this project came from reasoning about the budget
instead of measuring it.

Initial budget, 35.09 ms/token (ctx 32768 era):

| | share | what |
|---|---:|---|
| PCIe expert streaming | 37% | the obvious suspect |
| **dense weights in bf16** | **32.5%** | the adapter dequantized all 613 non-expert tensors |
| expert GEMV kernel | 10% | |
| everything else | ~20% | attention, sampling, host work |

**The ablation that redirected the whole effort:** making expert copies *free* — ablating
PCIe entirely — would have reached only 45.4 tok/s. So PCIe was not the ceiling, and
tuning the cache harder was not the answer. The dense weights were.

Final budget after the work: **13.77 ms of kernels**, with decode 97–99% inside the CUDA
graph, attention at 1.1% of the token and sampling at 0.6%. The dense GEMV runs at
305 GB/s against the card's 360 GB/s peak — i.e. it is now bandwidth-bound and there is
no arithmetic left to recover.

`FT_DECODE_PROF=1` (commit 12) is the instrument; it times the graph replay on the stream
and compares it with the step period, because a CUDA-graph decode hides everything outside
the graph from ordinary host-side timing.

---

## 2. The levers that paid

ctx 32768 era, in the order they were applied:

| lever | decode | why |
|---|---|---|
| **Dense weights stay packed** | 31.1 → **45.9** | 613 tensors were being materialized to bf16: 4.560 GiB instead of 2.380, and 3.613 GiB read per token instead of 1.876. Half the win is read bandwidth, half is the 2.18 GiB of VRAM that freed ~1100 expert cache slots. |
| **MMQ tile geometry** | prefill 907 → **1738** | vLLM's port of llama.cpp's MMQ kept the real per-architecture tiles behind `#if defined(USE_ROCM)`. Every CUDA build got `MMQ_X_Q4_K 4` instead of 64 and re-read each weight tile `ncols_y/mmq_x` times. `qkv_proj` went 2.28 → 16.0 TOP/s. |
| **MMQ instead of GEMV on prefill** | prefill 315 → 903 | Right kernel for the batch shape. |
| **Expert-bank stride from the tensor** | 45.9 → **47.6** | Removed row padding without touching a single weight; cache 3650 → 3750 slots. |
| **K-quant CPU dot products → `hybrid`** | 49.1 → **57.7** | The largest single win. See below. |

### Why `hybrid` was worth a C++ port

`--moe-backend hybrid` splits cache misses between PCIe fetch and CPU compute. It was
unreachable for GGUF because the CPU executor had no K-quant kernels. Porting ggml's
AVX2 `q4_K`/`q5_K`/`q6_K` dot products (plus `quantize_row_q8_K`) made it reachable.

**Measured CPU MoE bandwidth: 34.9 GB/s — 95% of this box's STREAM ceiling, and double
the 18–25 GB/s that was estimated.** K-quant dot never leaves integer SIMD
(`VPMADDUBSW` + `VPMADDWD`, scales as int16 multipliers, one `VFMADD` per 256 weights),
whereas NVFP4 dequantizes an e4m3 scale every 16 elements. The 256-wide superblock
amortizes scale work far better: **K-quants are *faster* on CPU than the format this
machinery was written for.**

### The tuned constants, and why those values

| knob | value | how it was chosen |
|---|---|---|
| MoE MMQ tiles | **32/128/4** | Swept. Not llama.cpp's dense answer (64 for Q4_K) — 64 loses to 32 here on routed-row padding. 4.01x over vLLM's 4/32/4. Re-sweep on other hardware with `benchmarks/gguf_mmq_tilesweep.py`. |
| `FREETOKEN_HYBRID_FETCH_FRACTION` | **0.28** | The profiled split (0.337) systematically over-fetches. Measured: 0.18→53.4, 0.25→57.5, **0.28→57.7**, 0.42→55.5. Per-checkpoint; sweep it. |
| `--moe-cache-size` | largest that survives **load**, minus one step | Smoke tests are not a gate: 2900 slots at ctx 131072 start, pass smoke 5/5 and die under concurrency. 2950 OOMs on graph capture. |
| `--memory-ratio` | **0.93**, not 0.95 | 0.95 leaves ~0.08–0.2 GiB free and dies mid-benchmark. |
| dense `_mm` dispatch | ≤6 tokens → MMVQ | MMQ beats GEMV for expert *banks* but loses to cuBLAS 8–13x for a *dense* projection. Both facts are true; they are about different alternatives. |

---

## 3. The levers that paid nothing

**This is the valuable half. Everything here was tried and measured — do not repeat it.**

Base for the table: ctx 131072, `--max-running-requests 1`, hybrid, fraction 0.28,
2850 slots, `--memory-ratio 0.95`. Baseline **decode 53.9 / prefill 1615**.

| lever | value | decode | prefill | verdict |
|---|---|---:|---:|---|
| `--attention-backend` | `fi` (auto) | **53.9** | **1615** | auto already wins |
| | `triton` | 51.9 | 1498 | works, frees VRAM, still loses |
| | `triton`, cache 3000 | 52.7 | 1516 | loses *even* after spending what it freed |
| | `fa` | — | — | `flash_fwd_launch_template.h:203 invalid argument` — Hopper kernels, not sm_86 |
| | `trtllm` | — | — | refused: "requires a compute capability 10.x GPU" |
| | `dsa`, `dsv4_sparse` | — | — | refused: "qwen3_5_moe uses full attention" |
| `--expert-load` | auto/serial/parallel | 53.9/53.6/53.7 | | noise; only affects startup bank reads |
| `--page-size` | 1 / 16 / 64 | **53.9**/51.5/51.0 | | −2.4 and −2.9 tok/s |
| `--cache-type` | radix / naive | **53.9**/52.3 | | naive also discards prefix reuse |
| `--kv-reserve-tokens` | 2048 | 53.7 | 1609 | inert with an explicit `--moe-cache-size` |
| `--cuda-graph-max-bs` | auto(=1) / 1 | 53.9/54.0 | | identical, as expected |
| | **0 (no graphs)** | **20.7** | 1602 | graphs are worth **2.6x** — never disable |
| `--decode-log-interval` | 40 / 1e6 | 53.9/54.3 | | noise; the log line is free |
| `--enable-cache-report` | on / off | 53.9/53.5 | | **free** — leave it on, it buys visibility |
| `UD-Q4_K_S` checkpoint | | 53.6 | 1615 | see below |
| GDN pool trim to 4 slots | | — | — | passes smoke, **dies under load** |
| `--moe-backend fused` | | — | — | refused, and inapplicable: 19.5 GiB of experts on a 12 GiB card |
| `--moe-backend cpu` | all experts on CPU | **30.8** | | a whole layer on CPU is ~0.8 ms vs 0.085 ms on GPU |
| `--moe-hybrid-max-fetch` | 0 / 1 / 2 | 31.1/50.8/48.5 | | wrong regulator — a *fraction* is needed |
| `--moe-cpu-threads` | 4 / 5 / 6 | 55.7/57.7/57.8 | | auto is already right |
| llama.cpp's `--n-cpu-moe` idea | | — | | wins for llama.cpp because it has no 7 GiB GPU expert cache; with one, per-layer CPU offload costs more than the PCIe it saves |
| `rba/FreeToken-ROCm` Triton GGUF kernels | | — | | rejected on **their author's own numbers**: 6.6–7.6 tok/s vs 93.6 for HIP/CUDA MMVQ. A fallback path for unsupported hardware, not a speedup |

**`FREETOKEN_MAMBA_SSM_DTYPE=float16` is the only one that pays and is still off**: 54.3
vs 53.9, by shrinking the GDN pool 491 → 251 MiB. Not enabled by default because the
engine documents "some precision cost on the long recurrence" and no long-dialogue quality
check was run.

### `UD-Q4_K_S` — where intuition lies

Its `down` is Q4_K (288 B/row) against Q5_K (352) in Q4_K_M, which *looks* like an 8%
smaller expert. It is not: `_bank_geometry` sizes the expert region by the **widest** ggml
type across layers, and Q6_K `down` exists on three layers of **both** checkpoints. Both
therefore allocate 2048 × 420 B per layer. Same 5.4 GiB pool, same free VRAM,
**53.6 vs 53.9**. Coarser quantization, for free.

---

## 4. Context is cheaper than it looks

Measured each time with the largest cache that survives:

| ctx | 32768 | 65536 | 98304 | 131072 |
|---|---:|---:|---:|---:|
| decode | 57.4 | 56.2 | 54.7 | **53.9** |

KV is 20 KiB/token here (2.5 GiB at 131072 ≈ 1040 slots), but the slot curve is flat at
this end — about **0.007 tok/s per slot**. So the full agent window costs **3.5 tok/s
(−6%)**. Do not reflexively shrink the context to buy decode.

**Always sweep decode *and* prefill together.** At 3800 slots (ctx 32768) decode is still
rising while prefill drops 22% — prefill transients stop fitting. A decode-only sweep
picks 3800 and silently pays for it.

---

## 5. Measurement traps

Each of these produced a wrong conclusion here before it was caught.

* **The benchmark does not measure decode speed.** It reports
  `completion_tokens / total request time`. Prefill costs **~1.1 s per chunk** regardless
  of how many new tokens follow, so even two tokens on a fully cached prefix pay it. Real
  decode speed is the engine's `gen throughput` line.
* **…but `gen throughput` is an interval metric, not a result.** One run prints 59.0, then
  6.94, then 0.48. Quoting a single sample of it as a target is how a phantom "61.9 tok/s"
  goalpost came to exist here. Do not compare against it; compare end-to-end, at equal
  context, on the same harness.
* **A cold expert cache reads 14.9 tok/s where the server sustains 33**; a repeated long
  prompt served from the radix cache "shows" 7564 tok/s of prefill. Warm up, and vary
  prompts.
* **Smoke tests are not a configuration gate.** `stress` with mixed prompt sizes is.
* **Higher tok/s ≠ correct.** The hybrid slot-0 bug (commit 4) produced fluent nonsense at
  60–65 tok/s against an honest 57.7.
* **A test that only reads expert 0 validates nothing about strides** — offset 0 is the
  same under every convention. Always exercise a non-zero index.
* **Leaked workers poison sweeps.** Orphaned multiprocessing children (~1.5 GiB each) held
  pinned memory across restarts and drove the box into swap, invalidating a whole sweep.

---

## 6. The ceiling, and what is left

**62.5 tok/s**, measured by ablating expert streaming entirely. At 53.9 (ctx 131072) that
is 86% of the ceiling; at 57.7 (ctx 32768) it is 92%. The remainder is residual expert
traffic, and there is no cheap lever left for it.

What is genuinely unaddressed:

* **`_prefill_routed` streams whole expert layers.** All 256 experts of all 40 layers for
  any prefill chunk — 19.4 GiB — because a full 2048-token chunk really does route to
  essentially every expert (2048 × top-8 over 256 experts reaches 255.7 of them in
  expectation). The cost is **per chunk, not per token**. A small-chunk path modelled on
  `deepseek_v4/moe.py::_prefill_routed` (which switches on
  `hidden_states.shape[0] * top_k >= num_experts` — a 32-token threshold for this
  checkpoint) is the obvious fix. It was designed and written here, **not verified, and
  reverted.** It targets the ~1.1 s chunk cost, not decode.
* **IQ-quants.** CUDA kernels exist in the vendored sources; the Python metadata and CPU
  dot products do not.
* **AVX-512 / NEON.** Only ggml's AVX2 branches were ported, with a scalar fallback.

---

## 7. Reproducing any of this

```bash
benchmarks/gguf_mmq_tilesweep.py        # re-sweep MMQ/MoE tiles for your GPU
benchmarks/gguf_moe_mmq_poison_repro.py # deterministic reproducer for the OOB write
FT_DECODE_PROF=1 ft serve ...           # per-step decode budget
ft bench bw                             # profile the PCIe/CPU split, then sweep around it
```

Hardware for every number here: RTX 3060 12 GB (SM 8.6), Ryzen 5 5500 (6C/12T, AVX2, no
AVX-512), 31 GiB RAM, weights on NVMe. A deliberately small box — the whole point of the
offload/hybrid path is a 35B-A3B model that does not fit.
