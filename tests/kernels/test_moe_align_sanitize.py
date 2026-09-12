"""``sanitize_moe_align_tail`` must leave nothing the ggml MMQ kernels can dereference.

See ``benchmarks/gguf_moe_mmq_poison_repro.py`` for the CUDA reproducer this guards.
"""

import pytest
import torch

pytest.importorskip("triton")

from freetoken.kernel.triton.moe_align import sanitize_moe_align_tail  # noqa: E402


def test_tail_past_npp_becomes_sentinel_and_negative_expert():
    block_size = 16
    ncols_dst = 32
    # What moe_align_block_size leaves behind: the used region written, the rest
    # whatever torch.empty picked up from a recycled allocator page.
    sorted_ids = torch.full((64,), -999_999, dtype=torch.int32)
    sorted_ids[:ncols_dst] = torch.arange(ncols_dst, dtype=torch.int32)
    expert_ids = torch.full((4,), 7, dtype=torch.int32)  # 7 is a VALID id: no guard fires
    expert_ids[:2] = torch.tensor([0, 1], dtype=torch.int32)
    npp = torch.tensor([ncols_dst], dtype=torch.int32)

    sorted_ids, expert_ids = sanitize_moe_align_tail(
        sorted_ids, expert_ids, npp, block_size, ncols_dst
    )

    assert torch.equal(sorted_ids[:ncols_dst], torch.arange(ncols_dst, dtype=torch.int32))
    # sentinel == ncols_dst, which the kernel epilogue already drops
    assert bool((sorted_ids[ncols_dst:] == ncols_dst).all())
    assert torch.equal(expert_ids[:2], torch.tensor([0, 1], dtype=torch.int32))
    # the kernel returns on a negative expert id before it reads any token offset
    assert bool((expert_ids[2:] == -1).all())


def test_no_negative_token_offset_survives():
    block_size = 4
    sorted_ids = torch.tensor([0, 1, 2, 3, -999_999, -12_345, 4, -7], dtype=torch.int32)
    expert_ids = torch.tensor([0, 3], dtype=torch.int32)
    npp = torch.tensor([4], dtype=torch.int32)

    sorted_ids, expert_ids = sanitize_moe_align_tail(
        sorted_ids, expert_ids, npp, block_size, 8
    )

    assert bool((sorted_ids >= 0).all())
    assert bool((expert_ids[1:] == -1).all())


def test_fully_used_buffer_is_left_alone():
    block_size = 8
    sorted_ids = torch.arange(16, dtype=torch.int32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int32)
    npp = torch.tensor([16], dtype=torch.int32)

    out_sorted, out_experts = sanitize_moe_align_tail(
        sorted_ids.clone(), expert_ids.clone(), npp, block_size, 16
    )

    assert torch.equal(out_sorted, sorted_ids)
    assert torch.equal(out_experts, expert_ids)


def test_result_dtype_is_preserved():
    """The buffers go straight back into the kernel, which reads them as int32."""
    sorted_ids, expert_ids = sanitize_moe_align_tail(
        torch.zeros(16, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        8,
        16,
    )
    assert sorted_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
