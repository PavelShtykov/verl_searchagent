# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agent loop worker for fully async training that supports multi-output agent loops.

``AgentLoopWorker`` assumes each agent loop returns a single ``AgentLoopOutput`` and produces one
row per input sample. Context-managed agent loops instead return ``list[AgentLoopOutput]`` (one
segment per context-reset boundary). This worker flattens those segments into individual rows and
tags each with its trajectory id (``agent_session_id``, the input row index) and segment order
(``agent_output_index``) so the trainer can recover per-trajectory grouping for advantage
computation. Single-output agent loops keep producing exactly one row per sample.
"""

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.fully_async_policy.multi_trajectory import OUTPUT_INDEX_KEY, SESSION_ID_KEY


class FullyAsyncAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker that fans out ``list[AgentLoopOutput]`` into per-segment rows."""

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        """Pad and post-process every segment of a (possibly multi-output) agent loop run."""
        outputs = output if isinstance(output, list) else [output]
        return [await super()._agent_loop_postprocess(segment, validate, **kwargs) for segment in outputs]

    def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
        # ``inputs`` is one (possibly empty) list of padded segments per input sample.
        # The per-sample input metadata is only consumed by the parent when reward is computed by
        # the trainer (no reward loop workers); otherwise expanding it is unused work, so skip it.
        expand_input = input_non_tensor_batch is not None and self.reward_loop_worker_handles is None

        flat_outputs = []
        session_ids = []
        output_indices = []
        source_rows = []
        for session_id, segments in enumerate(inputs):
            if not isinstance(segments, list):
                segments = [segments]
            for output_index, segment in enumerate(segments):
                flat_outputs.append(segment)
                session_ids.append(session_id)
                output_indices.append(output_index)
                source_rows.append(session_id)

        if not flat_outputs:
            raise RuntimeError("All agent loops returned empty outputs; nothing to post-process.")

        expanded_non_tensor = None
        if expand_input:
            expanded_non_tensor = {}
            for key, values in input_non_tensor_batch.items():
                array = np.empty(len(source_rows), dtype=object)
                for row, source in enumerate(source_rows):
                    array[row] = values[source]
                expanded_non_tensor[key] = array

        batch = super()._postprocess(flat_outputs, input_non_tensor_batch=expanded_non_tensor, validate=validate)
        batch.non_tensor_batch[SESSION_ID_KEY] = np.array(session_ids, dtype=np.int32)
        batch.non_tensor_batch[OUTPUT_INDEX_KEY] = np.array(output_indices, dtype=np.int32)
        return batch
