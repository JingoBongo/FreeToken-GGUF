"""Qwen3.5-MoE (llama.cpp arch ``qwen35moe``) GGUF adapter for FreeToken.

Serves a llama.cpp GGUF of Qwen3.6/Ornith-class MoE models directly -- no
requantization. The routed experts stay in their native packed blocks (they are ~32.2B
of the 35B parameters and cannot be materialized), and so does every dense tensor whose
ggml type has an MMVQ/MMQ kernel: attention qkv/o, the GDN input/output projections, the
shared expert, the LM head and the token embedding are served straight out of their
packed blocks by the borrowed ggml kernels. Only the tensors the GGUF itself stores as
F32 (norms, the router, ``ssm_alpha``/``ssm_beta``, ``conv1d``, the GDN recurrence
params) are materialized -- for those bf16 is *smaller* than the packed bytes.

Keeping the dense weights packed is worth 2.2 GiB of resident VRAM and ~5.5 ms/token of
read bandwidth; see ``FREETOKEN_GGUF_DENSE`` below for the per-group switch.

Provenance of every transform below: llama.cpp's own ``convert_hf_to_gguf.py``
(``Qwen3NextModel.modify_tensors`` + ``_LinearAttentionVReorderBase.modify_tensors``),
cross-checked by measuring a GGUF/safetensors pair of the same base model
(Qwen3.6-35B-A3B UD-Q4_K_M vs nvidia/Qwen3.6-35B-A3B-NVFP4). The two agree.

GGUF -> FreeToken transforms
----------------------------
* V-head order. HF groups the 32 value heads by key head ``(16, 2)``; ggml stores them
  tiled ``(2, 16)`` so ``ggml_repeat`` can replace an interleaved repeat. So
  ``gguf[i] == hf[P32[i]]`` with ``P32 = arange(32).reshape(16, 2).T.ravel()``; we invert
  it. Applied to: ``ssm_a``, ``ssm_dt.bias``, ``ssm_alpha``, ``ssm_beta``, the V rows of
  ``attn_qkv`` (rows >= 4096), all of ``attn_gate``, the V channels of ``ssm_conv1d``,
  and the *columns* of ``ssm_out``. Never to q/k rows or to full-attention layers.

  All of these survive in PACKED form. A row permutation is a plain ``index_select`` on
  the ``[out, row_bytes]`` block tensor. The one *column* permutation (``ssm_out``) works
  too because it permutes whole 128-element value heads and 128 is a whole number of quant
  blocks in every type here -- so it is a permutation of fixed-size byte groups
  (``head_dim // block * type_size``). ``verify/qcheck.py`` checks all of them
  bit-identically against dequantize-then-permute on the real checkpoint.
* ``ssm_a = -exp(A_log)`` -> ``A_log = log(-ssm_a)`` (bf16-exact on all 30 layers).
* ``conv1d`` was squeezed on the way out -> restore ``[8192, 1, 4]``.
* RMSNorm weights are stored as ``w`` here but as ``w - 1`` in HF (llama.cpp folds the
  +1 in; ``linear_attn.norm`` is excluded, being a plain norm). FreeToken's HF loader
  re-adds that 1, so its modules want ``w`` -- i.e. the GGUF value **unchanged**.
* Routed experts are NOT permuted and keep their expert order.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Iterator

import torch
import torch.nn.functional as F

from freetoken.layers.base import BaseOP
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_NAME,
    GGML_Q2_K,
    GGML_Q3_K,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen35moe"


# --------------------------------------------------------------------------------------
# Which dense tensors stay packed
# --------------------------------------------------------------------------------------
#
# ``FREETOKEN_GGUF_DENSE`` selects the groups (comma-separated; "all" = every group,
# "none"/"" = the original all-bf16 behaviour). It exists because each group was measured
# on its own -- see PERF.md -- and because it is the one-env-var way back to the previous
# behaviour if a checkpoint ever trips one of the packed paths.
_ALL_GROUPS = ("lmhead", "embed", "attn", "gdn", "shexp")

# role -> group. A "role" is one FreeToken module slot; its ggml type is recorded per
# layer in ModelConfig.dense_gguf_types at parse time (a Dynamic quant may vary it).
_ROLE_GROUP = {
    "lm_head": "lmhead",
    "embed": "embed",
    "qkv_proj": "attn",
    "o_proj": "attn",
    "in_proj_qkvz": "gdn",
    "out_proj": "gdn",
    "shexp_gate_up": "shexp",
    "shexp_down": "shexp",
}

# Types with both an MMVQ (decode GEMV) and an MMQ (prefill GEMM) kernel. Anything else
# (F32/F16 tensors: norms, router, ssm_alpha/beta, conv1d) is materialized to bf16, which
# for those is *smaller* than the packed bytes anyway.
_PACKABLE = frozenset(
    {GGML_Q4_0, GGML_Q8_0, GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K}
)


def _enabled_groups() -> frozenset[str]:
    raw = os.environ.get("FREETOKEN_GGUF_DENSE", "all").strip().lower()
    if raw in ("", "none", "0", "off"):
        return frozenset()
    if raw == "all":
        return frozenset(_ALL_GROUPS)
    want = {g.strip() for g in raw.split(",") if g.strip()}
    unknown = want - set(_ALL_GROUPS)
    if unknown:
        raise ValueError(
            f"FREETOKEN_GGUF_DENSE: unknown group(s) {sorted(unknown)}; "
            f"known groups are {list(_ALL_GROUPS)} (or 'all' / 'none')"
        )
    return frozenset(want)


class _DensePlan:
    """Which (role, layer) pairs are served packed, and with which ggml type.

    Built identically by the weight loader and by the module swap, so the emitted
    parameter names and the constructed modules cannot disagree (a disagreement would
    fail loudly at the strict ``load_state_dict``, but only after a 90 s load).
    """

    def __init__(self, config: ModelConfig):
        self.groups = _enabled_groups()
        self._types: dict[str, tuple[int, ...]] = dict(config.dense_gguf_types or ())
        # The GDN out_proj is the one role whose transform is a COLUMN permutation, and
        # that is only expressible on packed bytes when a value head is a whole number of
        # quant blocks (see _unpermute_packed_cols). Q8_0's block is 32, so a 128-wide
        # head always is; every K-quant's is 256, so it never is. Keep the rest of the
        # "gdn" group packed (in_proj_qkvz needs only ROW permutations, fine at any block)
        # and drop this single role to bf16 instead of refusing the checkpoint.
        self._v_head_dim = 0
        for grp in (getattr(config, "attention_groups", ()) or ()):
            v = getattr(grp, "value_head_dim", None)
            if v:
                self._v_head_dim = int(v)
                break

    def type_of(self, role: str, layer: int = 0) -> int | None:
        """The ggml type to keep ``role`` packed in, or ``None`` for the bf16 path."""
        if _ROLE_GROUP[role] not in self.groups:
            return None
        types = self._types.get(role)
        if types is None or layer >= len(types):
            return None
        tt = types[layer]
        if tt not in _PACKABLE:
            return None
        if role == "out_proj" and self._v_head_dim % BLOCK_SHAPE[tt][0] != 0:
            return None
        return tt

    def any_packed(self) -> bool:
        return bool(self.groups) and bool(self._types)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

def _expert_types(model_path: str) -> tuple[dict[int, int], dict[int, int]]:
    """(gate_up type, down type) per layer, read from the GGUF tensor table.

    Unsloth-Dynamic quants deliberately vary the expert type by layer (measured: a
    UD-Q4_K_M has Q4_K gate/up everywhere but Q5_K down on 37 layers and Q6_K on three),
    so the type is per layer, not per checkpoint. ``gate`` and ``up`` always share a type
    within a layer, which is what lets them live in one fused bank.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    gate_up: dict[int, int] = {}
    up_seen: dict[int, int] = {}
    down: dict[int, int] = {}
    pat = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
    for t in iter_gguf_tensors(model_path):
        m = pat.match(t.name)
        if not m:
            continue
        layer, kind = int(m.group(1)), m.group(2)
        if kind == "gate":
            gate_up[layer] = int(t.ggml_type)
        elif kind == "up":
            up_seen[layer] = int(t.ggml_type)
        else:
            down[layer] = int(t.ggml_type)
    for layer, tt in up_seen.items():
        if gate_up.get(layer) != tt:
            raise NotImplementedError(
                f"layer {layer}: ffn_gate_exps is "
                f"{GGML_NAME.get(gate_up.get(layer), gate_up.get(layer))} but ffn_up_exps is "
                f"{GGML_NAME.get(tt, tt)}; they share one fused bank and must match"
            )
    return gate_up, down


# GGUF tensor suffix -> (role, "the parts that must share a type"). Roles fused from
# several GGUF tensors can only stay packed when every part has the same ggml type: the
# fused qweight is one packed tensor of one type (see layers/gguf.py's docstring).
_ROLE_SOURCES = {
    "qkv_proj": ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
    "o_proj": ("attn_output.weight",),
    "in_proj_qkvz": ("attn_qkv.weight", "attn_gate.weight"),
    "out_proj": ("ssm_out.weight",),
    "shexp_gate_up": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    "shexp_down": ("ffn_down_shexp.weight",),
}


def _dense_types(model_path: str, num_layers: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Per-layer ggml type of every dense role, as a hashable ModelConfig payload.

    ``-1`` means "this layer has no such tensor" (a GDN layer has no ``attn_q``) or "the
    parts disagree on a type", which drops the role back to bf16 for that layer.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    seen: dict[tuple[str, int], int] = {}
    head: dict[str, int] = {}
    for t in iter_gguf_tensors(model_path):
        if t.name == "output.weight":
            head["lm_head"] = int(t.ggml_type)
            continue
        if t.name == "token_embd.weight":
            head["embed"] = int(t.ggml_type)
            continue
        if not t.name.startswith("blk."):
            continue
        parts = t.name.split(".", 2)
        layer, suffix = int(parts[1]), parts[2]
        seen[(suffix, layer)] = int(t.ggml_type)

    out: list[tuple[str, tuple[int, ...]]] = []
    for role, suffixes in _ROLE_SOURCES.items():
        per_layer = []
        for layer in range(num_layers):
            types = {seen.get((sfx, layer)) for sfx in suffixes}
            if None in types or len(types) != 1:
                per_layer.append(-1)  # absent on this layer, or a mixed-type fusion
            else:
                per_layer.append(types.pop())
        out.append((role, tuple(per_layer)))
    for role in ("lm_head", "embed"):
        if role in head:
            out.append((role, (head[role],)))
    return tuple(out)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str, default=None):
        val = m.get(f"{_ARCH}.{key}")
        if val is None:
            if default is not None:
                return default
            raise KeyError(f"missing GGUF metadata key {_ARCH}.{key}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    num_kv_heads = int(g("attention.head_count_kv"))
    head_dim = int(g("attention.key_length"))
    max_pos = int(g("context_length"))

    # llama.cpp writes rope.dimension_count = head_dim * partial_rotary_factor, which is
    # exactly the rotary width FreeToken wants (Qwen3.5 is partial-rotary, factor 0.25).
    rotary_dim = int(g("rope.dimension_count"))
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=float(g("rope.freq_base")),
        # Text-only: the mRoPE sections reduce to standard partial rope, and carrying the
        # unhashable mrope_section list into get_rope's cache key is what the HF path
        # avoids too.
        scaling=None,
    )

    interval = int(g("full_attention_interval", 4))
    full_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(num_layers) if (i + 1) % interval != 0)

    # GDN geometry. llama.cpp stores the Qwen3.5 linear-attention shape in generic SSM
    # keys (Qwen3NextModel.set_gguf_parameters): group_count = key heads,
    # time_step_rank = value heads, state_size = both head dims.
    num_k_heads = int(g("ssm.group_count"))
    num_v_heads = int(g("ssm.time_step_rank"))
    head_kv_dim = int(g("ssm.state_size"))

    gate_up_types, down_types = _expert_types(shim.model_path)
    gu_types = tuple(gate_up_types[i] for i in range(num_layers))
    dn_types = tuple(down_types[i] for i in range(num_layers))
    moe_i = int(g("expert_feed_forward_length"))
    _, gu_rows = _bank_geometry(hidden, 2 * moe_i, gu_types)
    _, dn_rows = _bank_geometry(moe_i, hidden, dn_types)

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,  # MoE: no dense decoder MLP
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        # The GGUF ships a separate output.weight, so the head is never tied.
        tie_word_embeddings=False,
        rotary_config=rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=False,
        moe_enabled=True,
        use_qk_norm=True,
        model_type="qwen3_5_moe",
        architectures=list(shim.architectures),
        vision_config=None,  # GGUF is text-only
        image_token_id=None,
        attention_groups=tuple(
            sorted(
                (
                    FullAttentionGroupConfig(
                        name="full",
                        layer_ids=full_ids,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        rotary_config=rotary,
                    ),
                    LinearGatedDeltaGroupConfig(
                        name="linear",
                        layer_ids=linear_ids,
                        num_key_heads=num_k_heads,
                        num_value_heads=num_v_heads,
                        key_head_dim=head_kv_dim,
                        value_head_dim=head_kv_dim,
                        conv_kernel_dim=int(g("ssm.conv_kernel")),
                        output_gate=True,
                    ),
                ),
                key=lambda grp: grp.layer_ids[0] if grp.layer_ids else 1 << 30,
            )
        ),
        # Routed experts stay in native ggml blocks; so does every dense tensor whose
        # ggml type has a kernel (see _DensePlan). The other quant knobs describe
        # safetensors quantization schemes and stay off.
        expert_quant="gguf_k",
        moe_weight_format="gguf_k",
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        moe_gguf_gate_up_types=gu_types,
        moe_gguf_down_types=dn_types,
        moe_gguf_gate_up_rows=gu_rows,
        moe_gguf_down_rows=dn_rows,
        dense_gguf_types=_dense_types(shim.model_path, num_layers),
    )


def is_gguf_model(config: ModelConfig) -> bool:
    return getattr(config, "moe_weight_format", None) == "gguf_k"


# --------------------------------------------------------------------------------------
# V-head order
# --------------------------------------------------------------------------------------

def _v_perm_inv(num_k_heads: int, num_v_heads: int, head_dim: int) -> torch.Tensor:
    """Index vector turning ggml's tiled V-head order back into HF's grouped order.

    ``hf = gguf[_v_perm_inv(...)]``. For ``head_dim > 1`` the head permutation is lifted
    onto the per-head blocks of rows.
    """
    r = num_v_heads // num_k_heads
    head_inv = torch.arange(num_v_heads).reshape(r, num_k_heads).t().reshape(-1)
    if head_dim == 1:
        return head_inv
    return (head_inv.unsqueeze(1) * head_dim + torch.arange(head_dim)).reshape(-1)


def _unpermute(t: torch.Tensor, dim: int, perm: torch.Tensor) -> torch.Tensor:
    return t.index_select(dim, perm.to(t.device)).contiguous()


def _unpermute_packed_cols(
    packed: torch.Tensor, ggml_type: int, group: int, perm_heads: torch.Tensor
) -> torch.Tensor:
    """Permute the *columns* of a packed ``[out, row_bytes]`` tensor by whole value heads.

    Block quantization packs each output row independently along the input dim, so a
    column permutation is only expressible on the packed bytes when it moves whole quant
    blocks: the permutation moves 128-element value heads, which is 4 blocks of Q*_0/Q8_0
    (block 32) but NOT a whole number of K-quant blocks (block 256) -- a K-quant head
    boundary falls in the middle of a block. ``_DensePlan.type_of`` therefore refuses to
    keep ``out_proj`` packed for a K-quant and this function never sees one; the assert
    below is the backstop. Checked bit-identical against dequantize-then-permute in
    ``verify/qcheck.py``.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert group % block == 0, (
        f"value head dim {group} is not a whole number of {GGML_NAME.get(ggml_type)} "
        f"blocks ({block}); the column permutation is not expressible on packed bytes"
    )
    head_bytes = group // block * type_size
    rows, rb = packed.shape
    n_heads = rb // head_bytes
    assert n_heads * head_bytes == rb and n_heads == perm_heads.numel(), (
        f"packed row {rb} B does not split into {perm_heads.numel()} x {head_bytes} B heads"
    )
    return (
        packed.view(rows, n_heads, head_bytes)
        .index_select(1, perm_heads.to(packed.device))
        .reshape(rows, rb)
        .contiguous()
    )


# --------------------------------------------------------------------------------------
# Native-GGUF dense modules
# --------------------------------------------------------------------------------------

# Above this token count the MMVQ GEMV stops winning (matches vLLM's and
# layers/gguf.py's heuristic; measured crossover on this box is 4-8 tokens).
_MMVQ_TOKENS = 6
# Largest bf16 slab the prefill path may materialize at once. The dense projections here
# are 50 MiB at most, so this only ever chunks the 970 MiB lm_head -- which is reached
# only if more than _MMVQ_TOKENS sequences prefill in one batch.
_DEQ_BUDGET = int(os.environ.get("FREETOKEN_GGUF_DEQ_MIB", "256")) * 2 ** 20


def _mm(x: torch.Tensor, qweight: torch.Tensor, quant_type: int) -> torch.Tensor:
    """``x @ dequant(qweight).T``, dispatched by token count.

    **Not** ``layers/gguf.py``'s ``fused_mul_mat_gguf``, and the difference is the whole
    point. That one sends anything past the GEMV threshold to ``ggml_mul_mat_a8`` (MMQ),
    which on this box is **8-13x slower than cuBLAS bf16** for these shapes
    (``perf-tools/dbench.py``: gdn ``in_proj`` at 2048 tokens, 45.3 ms MMQ vs 3.8 ms
    cuBLAS) -- routing attention through it cost 18% of prefill end to end. Dequantizing
    the weight into a transient and handing cuBLAS the bf16 lands within 6-8% of a
    resident bf16 weight (4.09 ms vs 3.81 ms) while the weight stays packed in VRAM,
    which is the entire reason for doing this.

    At <= ``_MMVQ_TOKENS`` tokens (every decode step) MMVQ wins on both counts: 0.084 ms
    vs 0.152 ms bf16 for that same projection, because it never materializes the weight.

    The dequant is chunked along the output dim so the transient stays bounded no matter
    how large the weight is. The kernels index the activation as a dense
    ``[tokens, in_features]`` block, so a non-contiguous view has to be materialized
    first; ``.contiguous()`` on an already-contiguous tensor returns ``self``.
    """
    from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_vec_a8

    out_features = qweight.shape[0]
    x = x.contiguous()
    tokens = x.shape[0]
    if tokens == 0:
        return x.new_empty((0, out_features))
    if tokens <= _MMVQ_TOKENS:
        return ggml_mul_mat_vec_a8(qweight, x, quant_type, out_features)

    block, type_size = BLOCK_SHAPE[quant_type]
    in_features = qweight.shape[1] // type_size * block
    rows_per_chunk = max(1, _DEQ_BUDGET // (in_features * x.element_size()))
    if rows_per_chunk >= out_features:
        w = ggml_dequantize(qweight, quant_type, out_features, in_features, x.dtype)
        return x @ w.T
    out = x.new_empty((tokens, out_features))
    for lo in range(0, out_features, rows_per_chunk):
        hi = min(lo + rows_per_chunk, out_features)
        w = ggml_dequantize(qweight[lo:hi], quant_type, hi - lo, in_features, x.dtype)
        out[:, lo:hi] = x @ w.T
    return out


class GGUFDenseLinear(BaseOP):
    """Bias-free Linear over a packed ``[out, row_bytes]`` GGUF weight (TP=1)."""

    def __init__(self, in_features: int, out_features: int, quant_type: int):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(
            out_features, row_bytes(in_features, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mm(x, self.qweight, self._quant_type)


class GGUFGdnInProj(BaseOP):
    """GDN fused input projection with a packed ``qkv|z`` half and a bf16 ``b|a`` half.

    ``b``/``a`` are ``[32, hidden]`` and the GGUF stores them **F32**, so they cannot join
    the packed fusion (one qweight is one ggml type) and would not pay for it anyway --
    bf16 is half the size of their F32 source. Splitting the projection is what the
    block-FP8 path already does (``gdn.py``'s ``in_proj_qkvz`` / ``in_proj_ba``); doing it
    inside one module instead keeps ``gdn.py`` untouched, at the cost of one concat whose
    traffic is ~0.2% of a prefill chunk and ~24 KB at decode.
    """

    def __init__(self, in_features: int, qkvz_out: int, ba_out: int, quant_type: int):
        self.in_features = in_features
        self._quant_type = quant_type
        self.qweight = torch.empty(
            qkvz_out, row_bytes(in_features, quant_type), dtype=torch.uint8
        )
        self.ba_weight = torch.empty(ba_out, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        qkvz = _mm(x, self.qweight, self._quant_type)
        return torch.cat([qkvz, F.linear(x, self.ba_weight)], dim=-1)


class GGUFLMHead(BaseOP):
    """Untied LM head over a packed GGUF ``output.weight`` (TP=1).

    Mirrors ``ParallelLMHead.forward`` at TP=1 (and ``Nvfp4LMHead``, which exists for the
    same reason on the NVFP4 checkpoint): slice to the last token per sequence at prefill,
    then the ggml GEMV/GEMM instead of a bf16 ``F.linear`` over a 970 MiB weight.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices]
        return _mm(x, self.qweight, self._quant_type)


def convert_qwen3_5_to_gguf(model, config: ModelConfig) -> None:
    """In place: swap the dense modules whose GGUF tensors stay packed.

    Left as dense bf16 in every configuration: all RMSNorms, the router (``mlp.gate``),
    ``shared_expert_gate``, ``conv1d``, ``A_log``/``dt_bias`` (fp32) and the GDN ``b``/``a``
    rows -- the GGUF stores every one of them F32, so bf16 *shrinks* them, and the
    recurrence params are precision-sensitive.
    """
    from freetoken.layers.gguf import GGUFEmbedding

    plan = _DensePlan(config)
    if not plan.any_packed():
        return

    inner = model.model
    tt = plan.type_of("embed")
    if tt is not None:
        inner.embed_tokens = GGUFEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_type=tt,
        )
    tt = plan.type_of("lm_head")
    if tt is not None:
        model.lm_head = GGUFLMHead(config.vocab_size, config.hidden_size, tt)

    hidden = config.hidden_size
    for layer_id, layer in enumerate(inner.layers.op_list):
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            tt = plan.type_of("qkv_proj", layer_id)
            if tt is not None:
                attn.qkv_proj = GGUFDenseLinear(hidden, sum(attn._qkv_split), tt)
            tt = plan.type_of("o_proj", layer_id)
            if tt is not None:
                attn.o_proj = GGUFDenseLinear(attn.qo_attn_dim, hidden, tt)
        else:
            gdn = layer.linear_attn
            tt = plan.type_of("in_proj_qkvz", layer_id)
            if tt is not None:
                gdn.in_proj = GGUFGdnInProj(
                    hidden, gdn.conv_dim + gdn.value_dim, 2 * gdn.num_v_heads, tt
                )
            tt = plan.type_of("out_proj", layer_id)
            if tt is not None:
                gdn.out_proj = GGUFDenseLinear(gdn.value_dim, hidden, tt)
        # Dense (non-MoE) Qwen3.x variants have no shared expert; their MLP carries the
        # same two attribute names but is fed by different GGUF tensors, so skip it.
        shexp = getattr(layer.mlp, "shared_expert", None)
        if shexp is None:
            continue
        tt = plan.type_of("shexp_gate_up", layer_id)
        if tt is not None:
            shexp.gate_up_proj = GGUFDenseLinear(
                hidden, 2 * config.shared_expert_intermediate_size, tt
            )
        tt = plan.type_of("shexp_down", layer_id)
        if tt is not None:
            shexp.down_proj = GGUFDenseLinear(
                config.shared_expert_intermediate_size, hidden, tt
            )


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------

_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)

# GGUF suffix -> FreeToken module-relative name, for tensors needing no transform beyond
# a dequantize. RMSNorm weights are passed through verbatim: llama.cpp already folded the
# +1 in, which is exactly what FreeToken's modules expect.
_PLAIN = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
}


def _to_bf16(t) -> torch.Tensor:
    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape)


def _to_f32(t) -> torch.Tensor:
    """For the two GDN recurrence params the engine keeps in fp32.

    ``gdn.py`` allocates ``A_log``/``dt_bias`` as ``torch.float32`` and
    ``gdn_kernels.py`` passes them to the fla kernel commented "already fp32 (stored
    fp32)", so handing over bf16 here does not raise -- the kernel just reads the buffer
    as fp32 and the model emits noise.
    """
    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32).reshape(t.shape)


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"qwen35moe GGUF {what} supports TP=1 only (packed expert banks are not sharded)"
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (FreeToken param name, tensor) for every non-expert tensor.

    A role the :class:`_DensePlan` keeps packed is emitted as ``<module>.qweight`` (uint8
    block bytes, any V-head permutation applied in packed space); everything else is
    dequantized to bf16 (fp32 for ``A_log``/``dt_bias``) exactly as before.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    assert not include_moe_experts, (
        "qwen35moe GGUF keeps routed experts in native ggml blocks and only supports the "
        "offload backend; they are loaded by load_gguf_k_expert_sources()."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(cached_load_hf_config(model_path))
    plan = _DensePlan(config)
    grp = next(
        g for g in config.attention_groups if isinstance(g, LinearGatedDeltaGroupConfig)
    )
    k_heads, v_heads = grp.num_key_heads, grp.num_value_heads
    v_dim, qk_rows = grp.value_head_dim, grp.key_head_dim * grp.num_key_heads * 2
    perm_rows = _v_perm_inv(k_heads, v_heads, v_dim)   # 4096-row axis
    perm_heads = _v_perm_inv(k_heads, v_heads, 1)      # 32-element axis

    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    gdn_buf: dict[int, dict[str, torch.Tensor]] = {}
    shexp_buf: dict[int, dict[str, torch.Tensor]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            if plan.type_of("embed") is not None:
                yield "model.embed_tokens.qweight", t.packed()
            else:
                yield "model.embed_tokens.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            if plan.type_of("lm_head") is not None:
                yield "lm_head.qweight", t.packed()
            else:
                yield "lm_head.weight", _to_bf16(t)
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if not name.startswith("blk."):
            raise ValueError(f"unmapped qwen35moe GGUF tensor: {name}")
        if any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            continue  # routed experts -> offload banks

        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"

        if suffix in _PLAIN:
            yield f"{base}.{_PLAIN[suffix]}", _to_bf16(t)
            continue

        if suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
            continue

        # --- full-attention o_proj (no permutation) ------------------------------------
        if suffix == "attn_output.weight":
            if plan.type_of("o_proj", layer) is not None:
                yield f"{base}.self_attn.o_proj.qweight", t.packed()
            else:
                yield f"{base}.self_attn.o_proj.weight", _to_bf16(t)
            continue

        # --- shared expert: gate|up fuse on the output-row axis -------------------------
        if suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            packed_t = plan.type_of("shexp_gate_up", layer)
            slot = "gate" if "gate" in suffix else "up"
            shexp_buf.setdefault(layer, {})[slot] = t.packed() if packed_t is not None \
                else _to_bf16(t)
            sb = shexp_buf[layer]
            if "gate" in sb and "up" in sb:
                merged = torch.cat([sb["gate"], sb["up"]], dim=0)
                key = "qweight" if packed_t is not None else "weight"
                yield f"{base}.mlp.shared_expert.gate_up_proj.{key}", merged
                del shexp_buf[layer]
            continue
        if suffix == "ffn_down_shexp.weight":
            if plan.type_of("shexp_down", layer) is not None:
                yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
            else:
                yield f"{base}.mlp.shared_expert.down_proj.weight", _to_bf16(t)
            continue

        # --- full-attention layers: q|k|v fuse -----------------------------------------
        if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            packed_t = plan.type_of("qkv_proj", layer)
            qkv_buf.setdefault(layer, {})[suffix[5]] = t.packed() if packed_t is not None \
                else _to_bf16(t)
            sb = qkv_buf[layer]
            if len(sb) == 3:
                merged = torch.cat([sb["q"], sb["k"], sb["v"]], dim=0)
                key = "qweight" if packed_t is not None else "weight"
                yield f"{base}.self_attn.qkv_proj.{key}", merged
                del qkv_buf[layer]
            continue

        # --- GDN ------------------------------------------------------------------------
        if suffix == "ssm_a":
            # ssm_a == -exp(A_log); invert, then undo the V-head tiling. Stays fp32.
            a = _to_f32(t)
            yield f"{base}.linear_attn.A_log", _unpermute(torch.log(-a), 0, perm_heads)
            continue
        if suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", _unpermute(_to_f32(t), 0, perm_heads)
            continue
        if suffix == "ssm_conv1d.weight":
            # [8192, 4]: q|k channels untouched, V channels re-ordered; restore the
            # singleton channel axis the converter squeezed out.
            w = _to_bf16(t)
            w = torch.cat([w[:qk_rows], _unpermute(w[qk_rows:], 0, perm_rows)], dim=0)
            yield f"{base}.linear_attn.conv1d.weight", w.unsqueeze(1).contiguous()
            continue
        if suffix == "ssm_out.weight":
            # out_proj consumes the V axis, so the permutation is on its COLUMNS -- doable
            # packed because it moves whole value heads (see _unpermute_packed_cols).
            ptype = plan.type_of("out_proj", layer)
            if ptype is not None:
                yield (
                    f"{base}.linear_attn.out_proj.qweight",
                    _unpermute_packed_cols(t.packed(), ptype, v_dim, perm_heads),
                )
            else:
                yield (
                    f"{base}.linear_attn.out_proj.weight",
                    _unpermute(_to_bf16(t), 1, perm_rows),
                )
            continue
        if suffix in ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"):
            # qkv|z can stay packed (row permutations only); b|a are F32 in the GGUF and
            # always go bf16, so the fusion splits in two when qkv|z is packed.
            packed_t = plan.type_of("in_proj_qkvz", layer)
            if suffix in ("attn_qkv.weight", "attn_gate.weight") and packed_t is not None:
                w = t.packed()
                if suffix == "attn_qkv.weight":
                    w = torch.cat([w[:qk_rows], _unpermute(w[qk_rows:], 0, perm_rows)], dim=0)
                    slot = "qkv"
                else:
                    w, slot = _unpermute(w, 0, perm_rows), "z"
            else:
                w = _to_bf16(t)
                if suffix == "attn_qkv.weight":
                    w = torch.cat([w[:qk_rows], _unpermute(w[qk_rows:], 0, perm_rows)], dim=0)
                    slot = "qkv"
                elif suffix == "attn_gate.weight":
                    w, slot = _unpermute(w, 0, perm_rows), "z"
                else:
                    slot = "b" if "beta" in suffix else "a"
                    w = _unpermute(w, 0, perm_heads)
            gdn_buf.setdefault(layer, {})[slot] = w
            sb = gdn_buf[layer]
            if len(sb) == 4:
                # FreeToken fuses GDN input projections in this exact order (_FUSIONS).
                if packed_t is not None:
                    yield (
                        f"{base}.linear_attn.in_proj.qweight",
                        torch.cat([sb["qkv"], sb["z"]], dim=0),
                    )
                    yield (
                        f"{base}.linear_attn.in_proj.ba_weight",
                        torch.cat([sb["b"], sb["a"]], dim=0),
                    )
                else:
                    yield (
                        f"{base}.linear_attn.in_proj.weight",
                        torch.cat([sb["qkv"], sb["z"], sb["b"], sb["a"]], dim=0),
                    )
                del gdn_buf[layer]
            continue

        raise ValueError(f"unmapped qwen35moe GGUF tensor: {name}")

    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not gdn_buf, f"incomplete GDN in_proj groups: {sorted(gdn_buf)}"
    assert not shexp_buf, f"incomplete shared-expert merges: {sorted(shexp_buf)}"


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native ggml blocks) for the offload cache
# --------------------------------------------------------------------------------------

def _bank_geometry(in_features: int, out_rows: int, types) -> tuple[int, tuple[int, ...]]:
    """Per-expert byte size shared by all layers, plus the ``nrows`` each layer passes.

    The offload cache requires ONE bank shape for every layer -- it asserts it and bakes
    the row size into a CUDA-graph-safe copy descriptor -- but Unsloth-Dynamic quants give
    different layers different ggml types, hence different row widths. So size the
    per-expert region at the WIDEST type present and let every layer write its own rows
    contiguously from the start of its region; a narrower layer leaves the tail unused.

    Every layer can then pass its REAL row count, because both expert kernels take the
    expert stride from the tensor: ``ggml_moe_a8`` always did (``W.stride(0)``) and
    ``ggml_moe_a8_vec`` now does too (``patch_bank_stride.py``). Upstream's GEMV derived
    it as ``expert * nrows * blocks_per_row`` and looked at no stride at all, which is
    what forced the previous scheme -- pad in ROWS to a size that divides evenly in every
    type present, ``lcm(352, 420) = 887040`` B here, and report an inflated ``nrows``.
    Those surplus rows were copied over PCIe, multiplied, and sliced off the output.

    On this checkpoint: ``down`` 887040 -> 860160 B, slot 2066688 -> 2039808 (-1.30%),
    and 2520 -> 2048 rows on the 37 Q5_K layers (2112 -> 2048 on the 3 Q6_K ones). On a
    quant whose ``down`` is split Q4_K/Q6_K the row saving is 3010 -> 2048 on half the
    layers. See PERF.md 9.
    """
    widths = sorted({row_bytes(in_features, tt) for tt in types})
    size = out_rows * widths[-1]
    return size, tuple(out_rows for _ in types)


def _expert_specs(config: ModelConfig) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    gu_size, _ = _bank_geometry(H, 2 * I, config.moe_gguf_gate_up_types)
    dn_size, _ = _bank_geometry(I, H, config.moe_gguf_down_types)
    # Flat bytes per expert: the row split is per layer (see _bank_geometry), so the bank
    # itself must not bake one in.
    return {
        "gate_up": ((E, gu_size, 1), torch.uint8),
        "down": ((E, dn_size, 1), torch.uint8),
    }


def _fill_expert_bank(bank: torch.Tensor, packed: torch.Tensor, rows: int, rb: int) -> None:
    """Write ``packed`` ([E*rows*rb] bytes) into ``bank`` ([E, rows, bank_rb]).

    Each expert's bytes go contiguously from the start of its region; any padding lands
    at the tail of the expert, never between its rows.
    """
    E = bank.shape[0]
    flat = bank.view(E, -1)
    flat[:, : rows * rb] = packed.reshape(E, rows * rb)


def load_gguf_k_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native ggml block bytes.

    ``gate_up`` is ``[E, 2I, gu_bytes]`` per layer (gate rows then up rows, matching
    ``silu_and_mul``) and ``down`` is ``[E, H, dn_bytes]``.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    hb = alloc_layer_banks(_expert_specs(config), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    gu_types, dn_types = config.moe_gguf_gate_up_types, config.moe_gguf_down_types

    # gate and up are separate GGUF tensors that share one fused bank, so a layer is only
    # complete once both halves have landed.
    half_seen: dict[int, set[str]] = {}
    seen_gu, seen_dn = set(), set()

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if t.name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
                which = "gate" if t.name.endswith("ffn_gate_exps.weight") else "up"
                rb = row_bytes(H, gu_types[layer])
                # Rows must be contiguous at the layer's OWN row width, so address the
                # bank as a flat per-expert region: gate at [0, I*rb), up right behind it
                # at [I*rb, 2I*rb). Indexing the [I, 2I) row slice instead would place up
                # at I*bank_rb, which differs from I*rb whenever this layer is padded.
                flat = banks["gate_up"][layer].view(E, -1)
                off = 0 if which == "gate" else I * rb
                flat[:, off:off + I * rb] = t.packed().reshape(E, I * rb)
                half = half_seen.setdefault(layer, set())
                half.add(which)
                if len(half) < 2:
                    continue
                seen_gu.add(layer)
            elif t.name.endswith("ffn_down_exps.weight"):
                rb = row_bytes(I, dn_types[layer])
                _fill_expert_bank(banks["down"][layer], t.packed(), H, rb)
                seen_dn.add(layer)
            else:
                continue
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    want = set(range(L))
    assert seen_gu == want and seen_dn == want, (
        f"missing expert layers: gate_up {sorted(want - seen_gu)}, down {sorted(want - seen_dn)}"
    )
    return banks


def dummy_gguf_k_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    hb = alloc_layer_banks(_expert_specs(config), config.num_layers)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "is_gguf_model",
    "convert_qwen3_5_to_gguf",
    "load_gguf_k_expert_sources",
    "dummy_gguf_k_expert_sources",
]
