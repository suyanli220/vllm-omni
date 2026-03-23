"""OmniGPUModelRunner — thin inheritance layer over v2 GPUModelRunner.

Injects ``OmniModelState`` via ``load_model`` and adds a
``finish_requests`` hook to clean up the intermediate buffer.

Key integration detail: Omni models (e.g. Qwen3-Omni Thinker) return a
tuple ``(text_hidden_states, captured_layer_dict)`` from ``forward()``.
The v2 ``GPUModelRunner`` expects a plain tensor.  We wrap the model's
``forward`` to intercept the tuple, store the auxiliary data (e.g.
``captured_layer_dict``) on ``_last_aux_output``, and return only the
tensor.  The aux data is then recombined with the tensor in
``OmniARModelRunner.sample_tokens`` to build the ``OmniOutput``.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)


class OmniGPUModelRunner(GPUModelRunner):
    """Thin layer over v2 ``GPUModelRunner`` for Omni lifecycle hooks."""

    model_state: OmniModelState
    _last_aux_output: Any

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        self._last_aux_output = None
        self.model_state = init_omni_model_state(self.vllm_config, self.model, self.encoder_cache, self.device)
        if hasattr(self.model, "make_omni_output"):
            self._wrap_model_forward()

    def _wrap_model_forward(self) -> None:
        """Wrap ``model.forward`` to intercept tuple returns.

        Stores the second element on ``self._last_aux_output`` and
        returns only the first (a tensor), making the model compatible
        with the v2 runner's tensor-only expectation and CUDA graph
        capture.
        """
        original_forward = self.model.forward
        runner = self

        @functools.wraps(original_forward)
        def _wrapped(**kwargs: Any) -> Any:
            output = original_forward(**kwargs)
            if isinstance(output, tuple) and len(output) == 2:
                first, second = output
                if isinstance(first, torch.Tensor):
                    runner._last_aux_output = second
                    return first
            runner._last_aux_output = None
            return output

        self.model.forward = _wrapped  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Request lifecycle: clean up intermediate buffer on finish
    # ------------------------------------------------------------------

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished = scheduler_output.finished_req_ids
        preempted = scheduler_output.preempted_req_ids
        all_done = finished | preempted if preempted else finished
        for req_id in all_done:
            idx = self.req_states.req_id_to_index.get(req_id)
            if idx is not None:
                self.model_state.remove_request(idx)
        super().finish_requests(scheduler_output)
