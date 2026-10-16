# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise mixed RAS trajectories and generator ownership on CUDA."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
    test_mixed_ras_preserves_seeded_trajectories_and_greedy_rng as _check_mixed_ras,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    "parameters,seeded_rows,use_host_params",
    [
        ("mixed", (0, 1, 2, 3), True),
        ("mixed", (0, 2), True),
        ("mixed", (0, 1, 2, 3), False),
        ("defaults", (0, 2), True),
        ("all_random", (0, 1, 2, 3), True),
    ],
)
def test_mixed_ras_cuda_trajectories(parameters, seeded_rows, use_host_params):
    _check_mixed_ras(parameters, seeded_rows, use_host_params=use_host_params, device="cuda")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_invalid_logits_invalidate_isolated_cuda_context():
    # A device assert poisons its process's CUDA context, so isolate the failure.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import torch
                from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
                    _make_sampling_metadata, _make_talker_model,
                )

                metadata = _make_sampling_metadata(output_token_ids=[[]])
                for name in ("temperature", "top_p", "top_k", "frequency_penalties",
                             "presence_penalties", "repetition_penalties"):
                    setattr(metadata, name, getattr(metadata, name).cuda())
                try:
                    _make_talker_model().sample(torch.full((1, 3), float("nan"), device="cuda"), metadata)
                    torch.accelerator.synchronize()
                except RuntimeError as error:
                    assert "device-side assert" in str(error).lower(), str(error)
                else:
                    raise AssertionError("invalid logits were accepted")

                try:
                    torch.ones(1, device="cuda")
                    torch.accelerator.synchronize()
                except RuntimeError as error:
                    assert "device-side assert" in str(error).lower(), str(error)
                    print("CUDA_CONTEXT_INVALIDATED")
                else:
                    raise AssertionError("device assertion unexpectedly left the context usable")
            """),
        ],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "0"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no finite logits" in result.stderr
    assert "CUDA_CONTEXT_INVALIDATED" in result.stdout


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("layout", ["default", "seeded", "mixed", "shared", "default_alias", "mixed_alias"])
def test_conditional_replacement_matches_native_rng(layout):
    from vllm_omni.model_executor.models.cosyvoice3.v1_sampling import commit_rejection_rng, conditional_sample

    device = torch.device("cuda", torch.accelerator.current_device_index())
    default = torch.cuda.default_generators[device.index]
    original = default.get_state()
    shared = torch.Generator(device=device).manual_seed(71)
    independent = torch.Generator(device=device).manual_seed(83)
    layouts = {
        "default": {},
        "seeded": {i: torch.Generator(device=device).manual_seed(100 + i) for i in range(4)},
        "mixed": {2: independent, 0: shared},
        "shared": {3: shared, 0: shared, 2: shared, 1: shared},
        "default_alias": {2: default, 0: default},
        "mixed_alias": {3: shared, 2: default, 0: shared},
    }
    generators = layouts[layout]
    unique = {id(g): g for g in [default, *generators.values()]}
    scores = torch.sin(torch.arange(4 * 8193, device=device).reshape(4, 8193).float())
    try:
        for flags in ([False, True, False, True], [True, False, True, True], [False] * 4):
            before = {key: g.get_state().clone() for key, g in unique.items()}
            picked = torch.tensor([0, 1, 2, 3], device=device)
            rejected = torch.tensor(flags, device=device)
            actual = conditional_sample(scores, picked.clone(), rejected, generators)
            # Scheduling the GPU draw must not speculatively advance any generator.
            assert all(torch.equal(g.get_state(), before[key]) for key, g in unique.items())
            commit_rejection_rng(flags, generators, scores.shape[1], device)
            actual_states = {key: g.get_state().clone() for key, g in unique.items()}
            for key, g in unique.items():
                g.set_state(before[key])
            rows = [i for i, active in enumerate(flags) if active]
            expected = picked.clone()
            if rows:
                weights = scores[rows].clone()
                weights[torch.arange(len(rows), device=device), picked[rows]] = -float("inf")
                probs = weights.softmax(1)
                noise = torch.empty_like(probs)
                if any(row not in generators for row in rows):
                    noise.exponential_()
                for compact, row in enumerate(rows):
                    if row in generators:
                        noise[compact].exponential_(generator=generators[row])
                expected[rows] = (probs / noise).argmax(1)
            assert torch.equal(actual, expected)
            assert all(torch.equal(g.get_state(), actual_states[key]) for key, g in unique.items())
    finally:
        default.set_state(original)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("seeded_rows", [(0, 1, 2, 3), (0, 2)])
def test_deferred_ras_trajectory_and_token_feedback(monkeypatch, seeded_rows):
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

    sample = CosyVoice3Model.sample

    def deferred(model, *args, **kwargs):
        output = sample(model, *args, **kwargs, defer_rng=True)
        model.finalize_deferred_sampling(output.sampled_token_ids)
        cached = model.consume_sampled_token_ids_cpu_override(output.sampled_token_ids)
        if cached is not None:
            assert cached == output.sampled_token_ids.tolist()
        return output

    monkeypatch.setattr(CosyVoice3Model, "sample", deferred)
    _check_mixed_ras("mixed", seeded_rows, use_host_params=True, device="cuda")
