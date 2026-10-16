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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("consumer", ["history", "output"])
def test_async_feedback_commits_rng_once_at_token_consumption(consumer):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
        _make_sampling_metadata,
        _make_talker_model,
    )
    from vllm_omni.worker.gpu_ar_model_runner import _SamplingFeedbackOutput

    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[0] * 10, [0] * 10])
    for name in ("temperature", "top_p", "top_k", "frequency_penalties", "presence_penalties", "repetition_penalties"):
        setattr(metadata, name, getattr(metadata, name).cuda())
    metadata.top_k.fill_(1)
    generator = torch.Generator(device="cuda").manual_seed(71)
    metadata.generators = {0: generator}
    default = torch.cuda.default_generators[torch.accelerator.current_device_index()]
    original = default.get_state()
    before = (generator.get_state(), default.get_state())
    logits = torch.tensor([[3.0, 1.0, 0.0]] * 2, device="cuda")
    try:
        reference = model.sample(logits.clone(), metadata).sampled_token_ids.cpu()
        # Match the runner's existing discarded-request rollback ordering.
        generator.set_offset(generator.get_offset() - 4)
        expected = (generator.get_state(), default.get_state())
        generator.set_state(before[0])
        default.set_state(before[1])
        output = model.sample(logits.clone(), metadata, defer_rng=True)
        generator.set_offset(generator.get_offset() - 4)
        # Pending feedback must retain ownership if the live batch is compacted.
        metadata.generators.clear()
        stream = torch.cuda.Stream()
        wrapped = AsyncGPUModelRunnerOutput(
            model_runner_output=SimpleNamespace(),
            sampled_token_ids=output.sampled_token_ids,
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=stream,
            vocab_size=3,
        )
        feedback = model.stage_deferred_sampling(stream, wrapped.async_copy_ready_event)
        result = _SamplingFeedbackOutput(wrapped, feedback)
        metadata.output_token_ids = [[-1], [-1]]
        batch = SimpleNamespace(
            sampling_metadata=metadata,
            sampled_token_ids_cpu=wrapped.sampled_token_ids_cpu,
            async_copy_ready_event=wrapped.async_copy_ready_event,
            prev_req_id_to_index={"a": 0, "b": 1},
            req_ids=["a", "b"],
        )
        if consumer == "history":
            InputBatch.update_async_output_token_ids(batch)
            model.commit_deferred_sampling()
            actual = result.get_output().sampled_token_ids
        else:
            actual = result.get_output().sampled_token_ids
            InputBatch.update_async_output_token_ids(batch)
            model.commit_deferred_sampling()
        feedback.commit()
        assert actual == reference.tolist() == metadata.output_token_ids
        assert torch.equal(generator.get_state(), expected[0])
        assert torch.equal(default.get_state(), expected[1])
        assert model._pending_rejection_rng is None
    finally:
        default.set_state(original)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("vocab", [2049, 6761, 8193, 9216])
def test_rejected_probability_preparation_matches_aten_bits(vocab):
    from vllm.triton_utils import triton

    from vllm_omni.model_executor.models.cosyvoice3.v1_sampling import _prepare_replacement_kernel
    from vllm_omni.utils.seeded_exponential import torch_exponential_policy

    scores = torch.sin(torch.arange(5 * vocab, device="cuda").reshape(5, vocab).float())
    scores[1].zero_()
    scores[2] = torch.linspace(-1000, 0, vocab, device="cuda")
    scores[3].fill_(-float("inf"))
    scores[3, 3] = 0
    picked = torch.arange(5, device="cuda")
    rejected = torch.tensor([True, True, True, True, False], device="cuda")
    snapshots = torch.zeros((5, 4), dtype=torch.int64, device="cuda")
    plan = torch.empty((5, 4), dtype=torch.int64, device="cuda")
    probs = torch.full_like(scores, float("nan"))
    threads, increment = torch_exponential_policy(vocab, scores.device)
    props = torch.cuda.get_device_properties(scores.device)
    _prepare_replacement_kernel[(5,)](
        rejected,
        snapshots,
        plan,
        scores,
        picked,
        probs,
        True,
        triton.next_power_of_2(vocab),
        5,
        vocab,
        8,
        threads,
        increment,
        props.multi_processor_count * (props.max_threads_per_multi_processor // 256) * 256,
        num_warps=32,
        enable_fp_fusion=False,
    )
    # Independent ATen oracle, including the sole-finite-candidate fallback.
    masked = scores.clone()
    selected = masked.gather(1, picked[:, None])
    masked.scatter_(1, picked[:, None], -float("inf"))
    masked.scatter_(
        1, picked[:, None], torch.where(torch.isfinite(masked).any(1, keepdim=True), -float("inf"), selected)
    )
    assert torch.equal(probs[:4], masked.softmax(1)[:4])
    assert torch.isnan(probs[4]).all()  # Accepted rows never prepare probabilities.
