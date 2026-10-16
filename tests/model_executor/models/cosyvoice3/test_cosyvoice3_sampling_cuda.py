# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise mixed RAS trajectories and generator ownership on CUDA."""

import os
import subprocess
import sys
import textwrap

import pytest

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
