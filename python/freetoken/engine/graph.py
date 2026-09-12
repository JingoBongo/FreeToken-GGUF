from __future__ import annotations

import gc
import sys
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


# ---------------------------------------------------------------- FT-DECODE-PROF
# Env-gated decode-step instrumentation. A CUDA-graph decode hides everything outside
# the graph from ordinary host-side timing, so the replay has to be timed on the stream
# and compared with the step period; the difference is the non-graph cost per token.
class _DecodeProf:
    def __init__(self, mode):
        import collections
        self.mode = mode
        self.every = int(os.environ.get("FT_DECODE_PROF_EVERY", "64"))
        self.skip = int(os.environ.get("FT_DECODE_PROF_SKIP", "48"))
        self.nprof = int(os.environ.get("FT_DECODE_PROF_N", "48"))
        self.out = os.environ.get(
            "FT_DECODE_PROF_OUT", os.path.expanduser("~/freetoken-decode-prof.txt"))
        self.n = 0
        self.pairs = collections.deque()
        self.prep_s = 0.0
        self.launch_s = 0.0
        self.period_s = 0.0
        self.period_n = 0
        self.last_entry = None
        self.gpu_ms = 0.0
        self.gpu_n = 0
        self.prof = None
        self.prof_done = False
        self.prof_started_at = None

    def _flush(self):
        import torch
        if not self.pairs:
            return
        torch.cuda.synchronize()
        while self.pairs:
            a, b = self.pairs.popleft()
            self.gpu_ms += a.elapsed_time(b)
            self.gpu_n += 1

    def _report(self):
        self._flush()
        if not self.gpu_n or not self.period_n:
            return
        gpu = self.gpu_ms / self.gpu_n
        period = self.period_s / self.period_n * 1e3
        prep = self.prep_s / self.gpu_n * 1e3
        launch = self.launch_s / self.gpu_n * 1e3
        outside = period - gpu
        sys.stderr.write(
            "FT-DECODE-PROF n=%d  period %.3f ms (%.1f tok/s)  graph-replay %.3f ms (%.1f%%)"
            "  outside-graph %.3f ms (%.1f%%)  [host prep %.3f, launch %.3f]\n"
            % (self.gpu_n, period, 1e3 / period if period else 0.0, gpu,
               100.0 * gpu / period if period else 0.0, outside,
               100.0 * outside / period if period else 0.0, prep, launch))
        sys.stderr.flush()
        self.gpu_ms = 0.0
        self.gpu_n = 0
        self.period_s = 0.0
        self.period_n = 0
        self.prep_s = 0.0
        self.launch_s = 0.0

    def _maybe_torch_profiler(self):
        if self.mode != "torch" or self.prof_done:
            return
        import torch
        if self.prof is None and self.n == self.skip:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False,
            )
            self.prof.__enter__()
            self.prof_started_at = self.n
            sys.stderr.write("FT-DECODE-PROF torch.profiler started at step %d\n" % self.n)
            sys.stderr.flush()
        elif self.prof is not None and self.n >= self.prof_started_at + self.nprof:
            torch.cuda.synchronize()
            self.prof.__exit__(None, None, None)
            steps = self.n - self.prof_started_at
            try:
                ka = self.prof.key_averages()
                with open(self.out, "w") as fh:
                    fh.write("decode steps profiled: %d\n\n" % steps)
                    fh.write(ka.table(sort_by="self_cuda_time_total", row_limit=90))
                    fh.write("\n\n==== by cpu self time ====\n\n")
                    fh.write(ka.table(sort_by="self_cpu_time_total", row_limit=40))
                sys.stderr.write("FT-DECODE-PROF torch.profiler table -> %s (%d steps)\n"
                                 % (self.out, steps))
            except Exception as exc:  # noqa: BLE001 -- must never kill a serve
                sys.stderr.write("FT-DECODE-PROF table failed: %r\n" % (exc,))
            sys.stderr.flush()
            self.prof = None
            self.prof_done = True

    def replay(self, runner, batch):
        import time

        import torch
        now = time.perf_counter()
        if self.last_entry is not None:
            self.period_s += now - self.last_entry
            self.period_n += 1
        self.last_entry = now
        self._maybe_torch_profiler()

        runner.buffer.copy_from(batch)
        g = runner.graph_map[batch.padded_size]
        runner.attn_backend.prepare_for_replay(batch)
        t1 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        g.replay()
        ev1.record()
        t2 = time.perf_counter()
        self.prep_s += t1 - now
        self.launch_s += t2 - t1
        self.pairs.append((ev0, ev1))
        self.n += 1
        if len(self.pairs) >= self.every:
            self._report()
        return runner.buffer.logits[: batch.size]


_FT_DECODE_PROF_MODE = os.environ.get("FT_DECODE_PROF", "").strip().lower()
_FT_DECODE_PROF = (
    _DecodeProf(_FT_DECODE_PROF_MODE)
    if _FT_DECODE_PROF_MODE in ("1", "events", "torch")
    else None
)
# -------------------------------------------------------------- /FT-DECODE-PROF


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if _FT_DECODE_PROF is not None:  # FT-DECODE-PROF
            return _FT_DECODE_PROF.replay(self, batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        gc.collect()
