# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise mixed RAS trajectories and generator ownership on CUDA."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
    _make_sampling_metadata,
    _make_talker_model,
)
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
    test_mixed_ras_preserves_seeded_trajectories_and_greedy_rng as _check_mixed_ras,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("parameters", ["mixed", "defaults", "all_random"])
@pytest.mark.parametrize("seeded_rows", [(0, 1, 2, 3), (0, 2)])
@pytest.mark.parametrize("use_host_params", [False, True])
def test_mixed_ras_cuda_trajectories(parameters, seeded_rows, use_host_params):
    _check_mixed_ras(parameters, seeded_rows, use_host_params=use_host_params, device="cuda")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_invalid_logits_raise_without_invalidating_cuda_context():
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    for name in ("temperature", "top_p", "top_k", "frequency_penalties", "presence_penalties", "repetition_penalties"):
        setattr(metadata, name, getattr(metadata, name).cuda())
    model = _make_talker_model()
    with pytest.raises(ValueError, match="no finite logits"):
        model.sample(torch.full((1, 3), float("nan"), device="cuda"), metadata)
    # A subsequent valid request must still be able to sample on this context.
    result = model.sample(torch.zeros(1, 3, device="cuda"), metadata)
    torch.accelerator.synchronize()
    assert result.sampled_token_ids.shape == (1, 1)
