# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 categorical draws with Torch-compatible Philox streams.

Both nucleus sampling and compacted RAS replacements use this primitive.
Probability construction stays in ATen to preserve nucleus boundaries. RNG
reservation stays on the host because V1 owns ordinary torch.Generators;
callers must compact rejected rows before requesting replacement draws.
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
    cols = tl.arange(0, block)
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
