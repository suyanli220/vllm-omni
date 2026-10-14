# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Categorical compatibility includes RNG state, not just token support."""

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.v1_sampling import categorical_sample, nucleus_sample
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("vocab", [3, 257, 6761])
@pytest.mark.parametrize("rows", [1, 4, 8, 65])
@pytest.mark.parametrize("seeding", ["all", "partial", "none", "shared", "default_alias"])
def test_categorical_matches_tokens_and_generator_states(vocab, rows, seeding):
    device = torch.device("cuda", 0)
    generators = {i: torch.Generator(device=device).manual_seed((1 << 63) + i) for i in range(rows)}
    if seeding == "partial":
        generators = {i: g for i, g in generators.items() if i % 2}
    elif seeding == "none":
        generators = {}
    elif seeding == "shared":
        generators = dict.fromkeys(range(rows), generators[0])
    elif seeding == "default_alias":
        generators = {0: torch.cuda.default_generators[0]}
    # Non-sorted dictionary order is observable for shared generators.
    generators = dict(reversed(list(generators.items())))
    torch.cuda.default_generators[0].manual_seed(42)
    probs = torch.randn(rows, vocab, generator=torch.Generator().manual_seed(17)).to(device).softmax(-1)
    owned = {id(g): g for g in [torch.cuda.default_generators[0], *generators.values()]}
    for step in range(20):
        before = {key: g.get_state() for key, g in owned.items()}
        expected = random_sample(probs.clone(), generators)
        after = {key: g.get_state() for key, g in owned.items()}
        for key, g in owned.items():
            g.set_state(before[key])
        actual = categorical_sample(probs, generators)
        assert torch.equal(actual, expected), (step, seeding)
        for key, g in owned.items():
            assert torch.equal(g.get_state(), after[key]), (step, seeding, key)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("vocab", [3, 257, 6761])
@pytest.mark.parametrize("top_k", [-1, 1, 25])
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("window", [0, 10])
def test_nucleus_exact_boundaries_ties_and_default_rng(vocab, top_k, seeded, window):
    device = torch.device("cuda", 0)
    # Equal weights make stable-sort ties and exact cumulative boundaries observable.
    probs, indices = torch.zeros(3, vocab, device=device).softmax(-1).sort(descending=True, stable=True)
    cumulative = probs.cumsum(-1)
    boundary = cumulative[0, 2] - probs[0, 2]
    top_p = torch.stack(
        [
            torch.nextafter(boundary, torch.full_like(boundary, -float("inf"))),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, float("inf"))),
        ]
    )
    limits = torch.full((3,), top_k, device=device, dtype=torch.int32)
    history = torch.tensor([[0] * window, [1] * window, [-1] * window], device=device, dtype=torch.long)
    generators = {i: torch.Generator(device=device).manual_seed(17 + i) for i in range(3)} if seeded else {}
    owned = [torch.cuda.default_generators[0], *generators.values()]
    for _ in range(5):
        before = [g.get_state() for g in owned]
        keep = (cumulative - probs) < top_p.unsqueeze(1)
        keep &= (limits.unsqueeze(1) <= 0) | (torch.arange(vocab, device=device) < limits.unsqueeze(1))
        draws = random_sample(probs * keep, generators).reshape(-1, 1)
        expected = indices.gather(1, draws).squeeze(1)
        repeated = ((history == expected.unsqueeze(1)).sum(1) >= 1.0) & (history >= 0).any(1)
        after = [g.get_state() for g in owned]
        for g, state in zip(owned, before):
            g.set_state(state)
        actual, actual_repeated = nucleus_sample(probs, indices, top_p, limits, history, 1.0, generators)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_repeated, repeated)
        for g, state in zip(owned, after):
            assert torch.equal(g.get_state(), state)
