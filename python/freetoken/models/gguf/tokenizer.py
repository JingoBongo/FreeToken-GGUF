"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key.
# qwen35moe has no converter of its own in transformers; its GGUF tokenizer is a
# gpt2-style BPE with the same tokens/merges/token_type fields the qwen3_moe
# converter reads.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "qwen35moe": "qwen3_moe"}


def _register_user_defined(fast, tok_dict: dict) -> tuple[int, int]:
    """Make the GGUF CONTROL (type 3) and USER_DEFINED (type 4) tokens match verbatim.

    transformers' GGUF converter registers only the handful it is told about by name, so
    the rest -- ``<think>``/``</think>``/``<tool_call>``/``</tool_call>``, which the chat
    template writes as literal text, and the vision/fim control markers -- were BPE-split
    into ordinary pieces instead of their real ids.

    CONTROL goes in as ``special=True`` (a marker should disappear under
    ``skip_special_tokens``), USER_DEFINED as ``special=False`` (the reasoning and
    tool-call parsers read those tags out of the decoded text). Returns the counts.
    """
    from tokenizers import AddedToken

    tokens = tok_dict.get("tokens") or []
    ttypes = tok_dict.get("token_type") or []
    if not tokens or len(ttypes) != len(tokens):
        return 0, 0
    vocab = fast.get_vocab()
    want = {True: [], False: []}
    for tid, (text, tt) in enumerate(zip(tokens, ttypes)):
        tt = int(tt)
        if tt not in (3, 4):
            continue
        # Only tokens already in the vocab at their own id: add_tokens would otherwise
        # APPEND a new id past the embedding table and corrupt every later encode.
        if vocab.get(text) != tid:
            continue
        if len(fast.encode(text, add_special_tokens=False).ids) == 1:
            continue  # already matches verbatim
        want[tt == 3].append(AddedToken(text, normalized=False, special=(tt == 3)))
    if not (want[True] or want[False]):
        return 0, 0
    before = fast.get_vocab_size(with_added_tokens=True)
    if want[True]:
        fast.add_special_tokens(want[True])
    if want[False]:
        fast.add_tokens(want[False])
    after = fast.get_vocab_size(with_added_tokens=True)
    assert after == before, (
        "registering GGUF special tokens grew the vocab %d -> %d; they must map to their "
        "existing ids" % (before, after)
    )
    return len(want[True]), len(want[False])


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)
    _register_user_defined(fast, tok_dict)

    tokens = tok_dict["tokens"]

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id).
    for name in ("<eos>", "<turn|>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
