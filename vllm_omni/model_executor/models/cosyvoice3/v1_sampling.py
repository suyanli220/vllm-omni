# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 categorical draws with Torch-compatible Philox streams.

Both nucleus sampling and compacted RAS replacements use this primitive.
Probability construction stays in ATen to preserve nucleus boundaries. RNG
reservation normally stays on the host because V1 owns torch.Generators.
The synchronous runner can defer replacement reservations until token feedback.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.seeded_exponential import torch_exponential_noise, torch_exponential_policy


@triton.jit
def _categorical_kernel(
    probs,
    state,
    out,
    cumulative,
    indices,
    top_p,
    top_k,
    history,
    rejected,
    threshold,
    replacement_plan,
    active,
    window: tl.constexpr,
    history_block: tl.constexpr,
    nucleus: tl.constexpr,
    vocab: tl.constexpr,
    stride: tl.constexpr,
    row_threads: tl.constexpr,
    batch_threads: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    if active is not None:
        if not tl.load(active + row):
            return
    cols = tl.arange(0, block)
    if replacement_plan is not None:
        seed = tl.load(replacement_plan + row * 4).to(tl.uint64)
        offset = tl.load(replacement_plan + row * 4 + 1).to(tl.uint64)
        threads = tl.load(replacement_plan + row * 4 + 2)
        start = tl.load(replacement_plan + row * 4 + 3)
        noise = torch_exponential_noise(seed, offset, start + cols, threads)
    else:
        seed = tl.load(state + row * 3).to(tl.uint64)
        offset = tl.load(state + row * 3 + 1).to(tl.uint64)
        if tl.load(state + row * 3 + 2):
            noise = torch_exponential_noise(seed, offset, cols, row_threads)
        else:
            noise = torch_exponential_noise(seed, offset, row * vocab + cols, batch_threads)
    weights = tl.load(probs + row * stride + cols, cols < vocab, other=0.0)
    if nucleus:
        cdf = tl.load(cumulative + row * vocab + cols, cols < vocab, other=0.0)
        p = tl.load(top_p + row)
        k = tl.load(top_k + row)
        keep = ((cdf - weights) < p) & ((k <= 0) | (cols < k))
        weights = weights * keep.to(tl.float32)
    # Match ATen float32 division, including ties at the last bit.
    scores = tl.where(cols < vocab, tl.div_rn(weights, noise), -float("inf"))
    selected = tl.argmax(scores, axis=0, tie_break_left=True)
    if nucleus:
        selected = tl.load(indices + row * vocab + selected)
    tl.store(out + row, selected)
    if window > 0:
        positions = tl.arange(0, history_block)
        tokens = tl.load(history + row * window + positions, positions < window, other=-1)
        repeats = tl.sum((tokens == selected).to(tl.int32), axis=0)
        has_history = tl.sum((tokens >= 0).to(tl.int32), axis=0) > 0
        tl.store(rejected + row, (repeats >= threshold) & has_history)


def _cuda_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    *,
    nucleus: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    history: torch.Tensor | None = None,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reserve exactly the draws that Torch would consume, then fuse selection.

    A partially seeded draw reserves one default-generator *matrix* first,
    including overwritten seeded rows, followed by request draws in dict order.
    """
    rows, vocab = probs.shape
    window = history.shape[1] if history is not None else 0
    out = torch.empty(rows, dtype=torch.long, device=probs.device)
    rejected = torch.empty(rows, dtype=torch.bool, device=probs.device) if history is not None else None
    if rejected is not None and window == 0:
        rejected.zero_()
    if rows == 0:
        return out, rejected
    if vocab == 0 or probs.stride(1) != 1:
        raise ValueError("sampling requires nonempty contiguous vocabulary rows")
    if any(row < 0 or row >= rows for row in generators):
        raise ValueError("generator row outside sampling batch")

    row_threads, row_increment = torch_exponential_policy(vocab, probs.device)
    batch_threads, batch_increment = torch_exponential_policy(rows * vocab, probs.device)

    def reserve(generator: torch.Generator, increment: int) -> tuple[int, int]:
        seed, offset = generator.initial_seed(), generator.get_offset()
        generator.set_offset(offset + increment)
        return seed if seed < 1 << 63 else seed - (1 << 64), offset

    # Per-row (seed, offset, row-draw flag). Default draws index the full matrix.
    states = [(0, 0, 0)] * rows
    if len(generators) != rows:
        default = torch.cuda.default_generators[probs.device.index or 0]
        states = [(*reserve(default, batch_increment), 0)] * rows
    for row, generator in generators.items():
        states[row] = (*reserve(generator, row_increment), 1)
    # Neither changing offsets nor seeded-row membership specialize the kernel.
    state = torch.tensor(states, dtype=torch.int64, pin_memory=True).to(probs.device, non_blocking=True)
    cumulative, indices, top_p, top_k = nucleus if nucleus is not None else (None, None, None, None)
    _categorical_kernel[(rows,)](
        probs,
        state,
        out,
        cumulative,
        indices,
        top_p,
        top_k,
        history,
        rejected,
        threshold,
        None,
        None,
        window=window,
        history_block=triton.next_power_of_2(max(window, 1)),
        nucleus=nucleus is not None,
        vocab=vocab,
        stride=probs.stride(0),
        row_threads=row_threads,
        batch_threads=batch_threads,
        block=triton.next_power_of_2(vocab),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, rejected


def _use_triton(probs: torch.Tensor) -> bool:
    # Bound supported batch sizes and per-program vocabulary storage.
    return (
        current_omni_platform.is_cuda()
        and probs.is_cuda
        and probs.dtype == torch.float32
        and probs.shape[0] <= 64
        and probs.shape[1] <= 65536
    )


def categorical_sample(probs: torch.Tensor, generators: dict[int, torch.Generator]) -> torch.Tensor:
    """Return an int64 token vector; replacements supply only rejected rows."""
    if not _use_triton(probs):
        return random_sample(probs, generators).reshape(-1).long()
    return _cuda_sample(probs, generators)[0]


def nucleus_sample(
    probs: torch.Tensor,
    indices: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    history: torch.Tensor | None,
    threshold: float,
    generators: dict[int, torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Filter sorted probabilities, draw, map token ids and check RAS history.

    ATen owns the stable sort, softmax and cumulative sum. Fusing pointwise
    masks and reductions does not change the full-distribution top-p boundary.
    """
    cumulative = probs.cumsum(dim=1)
    if _use_triton(probs):
        return _cuda_sample(
            probs, generators, nucleus=(cumulative, indices, top_p, top_k), history=history, threshold=threshold
        )
    keep = (cumulative - probs) < top_p.unsqueeze(1)
    ranks = torch.arange(probs.shape[1], device=probs.device)
    keep &= (top_k.unsqueeze(1) <= 0) | (ranks.unsqueeze(0) < top_k.unsqueeze(1))
    draws = random_sample(probs * keep.to(probs.dtype), generators).reshape(-1, 1).long()
    sampled = indices.gather(1, draws).squeeze(1)
    repeated = None
    if history is not None:
        repeated = ((history == sampled.unsqueeze(1)).sum(dim=1) >= threshold) & (history >= 0).any(dim=1)
    return sampled, repeated


@triton.jit
def _replacement_plan_kernel(
    rejected,
    snapshots,
    plan,
    rows: tl.constexpr,
    vocab: tl.constexpr,
    block: tl.constexpr,
    row_threads: tl.constexpr,
    row_increment: tl.constexpr,
    max_threads: tl.constexpr,
):
    row = tl.program_id(0)
    if not tl.load(rejected + row):
        return
    positions = tl.arange(0, block)
    valid = positions < rows
    active = tl.load(rejected + positions, valid, other=0)
    groups = tl.load(snapshots + positions * 4 + 2, valid, other=-1)
    seeded = tl.load(snapshots + positions * 4 + 3, valid, other=0)
    count = tl.sum(active.to(tl.int32), 0)
    rank = tl.sum((active & (positions < row)).to(tl.int32), 0)
    # Torch reserves a matrix for *all* compacted rows if any is unseeded.
    batch_threads = tl.minimum(max_threads, tl.cdiv(count * vocab, 256) * 256)
    batch_increment = tl.cdiv(count * vocab, batch_threads * 4) * 4
    uses_default = tl.sum((active & (seeded == 0)).to(tl.int32), 0) > 0
    group = tl.load(snapshots + row * 4 + 2)
    is_seeded = tl.load(snapshots + row * 4 + 3) != 0
    offset = tl.load(snapshots + row * 4 + 1)
    if is_seeded:
        preceding = tl.sum((active & (seeded != 0) & (groups == group) & (positions < row)).to(tl.int32), 0)
        offset += preceding * row_increment + tl.where((group == 0) & uses_default, batch_increment, 0)
    tl.store(plan + row * 4, tl.load(snapshots + row * 4))
    tl.store(plan + row * 4 + 1, offset)
    tl.store(plan + row * 4 + 2, tl.where(is_seeded, row_threads, batch_threads))
    tl.store(plan + row * 4 + 3, tl.where(is_seeded, 0, rank * vocab))


def conditional_sample(
    scores: torch.Tensor,
    sampled: torch.Tensor,
    rejected: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample rejected rows without advancing host generators until feedback.

    Snapshots follow the initial draw. Group zero is the default generator;
    compacted request draws follow ascending row order, including aliases.
    """
    rows, vocab = scores.shape
    default = torch.cuda.default_generators[scores.device.index or 0]
    groups = {id(default): 0}
    snapshots = []
    for row in range(rows):
        generator = generators.get(row, default)
        group = groups.setdefault(id(generator), len(groups))
        seed = generator.initial_seed()
        snapshots.append(
            (seed if seed < 1 << 63 else seed - (1 << 64), generator.get_offset(), group, row in generators)
        )
    state = torch.tensor(snapshots, dtype=torch.int64, pin_memory=True).to(scores.device, non_blocking=True)
    plan = torch.empty((rows, 4), dtype=torch.int64, device=scores.device)
    row_threads, row_increment = torch_exponential_policy(vocab, scores.device)
    props = torch.cuda.get_device_properties(scores.device)
    _replacement_plan_kernel[(rows,)](
        rejected,
        state,
        plan,
        rows,
        vocab,
        triton.next_power_of_2(rows),
        row_threads,
        row_increment,
        props.multi_processor_count * (props.max_threads_per_multi_processor // 256) * 256,
    )
    # Preserve ATen's probability construction, including the sole-candidate fallback.
    scores = scores.clone()
    ids = sampled.unsqueeze(1)
    selected = scores.gather(1, ids)
    scores.scatter_(1, ids, float("-inf"))
    scores.scatter_(1, ids, torch.where(torch.isfinite(scores).any(1, keepdim=True), float("-inf"), selected))
    probs = scores.softmax(1)
    _categorical_kernel[(rows,)](
        probs,
        None,
        sampled,
        None,
        None,
        None,
        None,
        None,
        None,
        0.0,
        plan,
        rejected,
        window=0,
        history_block=1,
        nucleus=False,
        vocab=vocab,
        stride=probs.stride(0),
        row_threads=0,
        batch_threads=0,
        block=triton.next_power_of_2(vocab),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return sampled


def commit_rejection_rng(
    rejected: list[bool],
    generators: dict[int, torch.Generator],
    vocab: int,
    device: torch.device,
) -> None:
    """Commit the same reservations as a native compacted replacement draw."""
    rows = [row for row, active in enumerate(rejected) if active]
    if not rows:
        return
    if any(row not in generators for row in rows):
        default = torch.cuda.default_generators[device.index or 0]
        increment = torch_exponential_policy(len(rows) * vocab, device)[1]
        default.set_offset(default.get_offset() + increment)
    increment = torch_exponential_policy(vocab, device)[1]
    for row in rows:
        if row in generators:
            generator = generators[row]
            generator.set_offset(generator.get_offset() + increment)
