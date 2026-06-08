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
"""Advantage computation for agent loops that emit multiple outputs per trajectory.

Context-managed agent loops (``AgentLoopWithContextManagement``) split a single rollout into
several ``AgentLoopOutput`` segments, one per context-reset boundary. The segments of one
trajectory share a single episode reward, so GRPO advantages must be computed once on the final
segment of each trajectory and broadcast to the remaining segments, mirroring the synchronous
``main_ppo_sync`` trainer.

Trajectories are identified by ``(uid, agent_session_id)`` and the segment order within a
trajectory by ``agent_output_index``; both columns are attached by ``FullyAsyncAgentLoopWorker``.
"""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage

SESSION_ID_KEY = "agent_session_id"
OUTPUT_INDEX_KEY = "agent_output_index"
IS_PADDING_KEY = "is_padding"


def pad_dataproto_to_multiple(batch: DataProto, multiple: int) -> DataProto:
    """Pad a batch with cheap masked no-op rows so its size is a multiple of ``multiple``.

    Multi-output rollouts emit a variable row count that need not divide the actor mini-batch size
    (``ppo_mini_batch_size * rollout.n``, itself a multiple of the data-parallel size). Padding to
    that multiple early — before seqlen balancing and the dp-sharded forward/backward passes — keeps
    every downstream step well-defined for data parallel > 1.

    Padding rows clone the first row (keeping every tensor shape/dtype valid) but keep a single
    attention token and zero the loss-bearing fields, so they add no gradient, no reward and
    negligible compute. Every row gets an ``is_padding`` flag so metrics / dumps can drop the
    synthetic rows; padding rows use a dedicated ``uid`` so they form their own (ignored) GRPO group
    and never perturb a real prompt's group. ``pad_size`` is always smaller than ``multiple``.
    """
    size = len(batch)
    batch.non_tensor_batch[IS_PADDING_KEY] = np.zeros(size, dtype=bool)
    remainder = size % multiple
    if remainder == 0:
        return batch

    pad_size = multiple - remainder
    pad = batch.select_idxs([0] * pad_size)
    if "attention_mask" in pad.batch.keys():
        attention_mask = torch.zeros_like(pad.batch["attention_mask"])
        attention_mask[:, 0] = 1  # keep one valid token so the packed forward stays well-formed
        pad.batch["attention_mask"] = attention_mask
    for key in ("response_mask", "loss_mask", "advantages", "returns", "token_level_scores", "token_level_rewards"):
        if key in pad.batch.keys():
            pad.batch[key] = torch.zeros_like(pad.batch[key])
    if "uid" in pad.non_tensor_batch:
        pad.non_tensor_batch["uid"] = np.array(["__pad__"] * pad_size, dtype=object)
    if SESSION_ID_KEY in pad.non_tensor_batch:
        pad.non_tensor_batch[SESSION_ID_KEY] = np.arange(pad_size, dtype=np.int32)
    if OUTPUT_INDEX_KEY in pad.non_tensor_batch:
        pad.non_tensor_batch[OUTPUT_INDEX_KEY] = np.zeros(pad_size, dtype=np.int32)
    pad.non_tensor_batch[IS_PADDING_KEY] = np.ones(pad_size, dtype=bool)

    # ``select_idxs`` carries ``batch.meta_info``; clear it so ``DataProto.concat`` does not assert
    # equality on array-valued meta_info entries (e.g. ``global_token_num``).
    pad.meta_info = {}
    padded = DataProto.concat([batch, pad])
    if "global_token_num" in padded.meta_info and "attention_mask" in padded.batch.keys():
        padded.meta_info["global_token_num"] = padded.batch["attention_mask"].sum(dim=-1).tolist()
    return padded


def has_multi_trajectory_outputs(batch: DataProto) -> bool:
    """Whether ``batch`` carries the per-segment bookkeeping emitted for multi-output agent loops."""
    return SESSION_ID_KEY in batch.non_tensor_batch and OUTPUT_INDEX_KEY in batch.non_tensor_batch


def compute_multi_trajectory_advantage(
    data: DataProto,
    *,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> DataProto:
    """Compute GRPO advantages from the final segment of each trajectory and broadcast them.

    Only the final segment of every ``(uid, agent_session_id)`` trajectory participates in the
    group-relative computation; the resulting per-trajectory advantage is then written to all
    segments of that trajectory. Non-GRPO estimators are delegated to :func:`compute_advantage`
    unchanged. For single-segment rollouts this reduces exactly to the standard GRPO computation.
    """
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        return compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    uids = data.non_tensor_batch["uid"]
    session_ids = data.non_tensor_batch[SESSION_ID_KEY]
    output_indices = data.non_tensor_batch[OUTPUT_INDEX_KEY]

    # Final (largest output index) row of each trajectory, keyed by (uid, session id).
    final_row_of_trajectory: dict[tuple, int] = {}
    trajectory_of_row: list[tuple] = []
    for row, (uid, session_id, output_index) in enumerate(zip(uids, session_ids, output_indices, strict=True)):
        trajectory_key = (uid, int(session_id))
        trajectory_of_row.append(trajectory_key)
        incumbent = final_row_of_trajectory.get(trajectory_key)
        if incumbent is None or output_indices[incumbent] < output_index:
            final_row_of_trajectory[trajectory_key] = row

    final_rows = list(final_row_of_trajectory.values())
    trajectory_to_local = {key: local for local, key in enumerate(final_row_of_trajectory)}
    row_to_local = np.fromiter(
        (trajectory_to_local[key] for key in trajectory_of_row), dtype=np.int64, count=len(trajectory_of_row)
    )

    # GRPO over the final segments only; groups stay keyed by uid (one final segment per trajectory).
    final_data = compute_advantage(
        data.select_idxs(final_rows),
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    first_response_token = final_data.batch["response_mask"].argmax(dim=1)
    trajectory_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_response_token]

    # Broadcast each trajectory's scalar advantage to all of its segments, masked to response tokens.
    broadcast_scores = trajectory_scores[torch.from_numpy(row_to_local)].unsqueeze(-1) * data.batch["response_mask"]
    data.batch["advantages"] = broadcast_scores
    data.batch["returns"] = broadcast_scores
    return data
