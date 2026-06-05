# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SearchAgent context manager — triggers context reset after successful save_and_advance.

Detection: SearchAgentSession sets a `pending_compression` flag in commit_save_and_advance
and exposes it via get_state_for_context(). This manager reads the flag from
state.extra_fields["session_state"] — a structural signal rather than text matching.

Compression: preserves the original prompt (system + user query + tool schemas)
and replaces all assistant/tool turns with a single assistant message containing
a structured round summary (persistent_saved chunks, prior round summaries, concept lists).
"""

import json
import logging
from typing import Any, Optional

from verl.experimental.agent_loop.context_manager import ContextManager, ContextState
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.tokenizer import normalize_token_ids

from .searchagent_session import SearchAgentSession

logger = logging.getLogger(__name__)


class SearchAgentContextManager(ContextManager):
    """Compresses context after successful save_and_advance, preserving cross-round state."""

    def __init__(
        self,
        tokenizer: Any,
        apply_chat_template_kwargs: Optional[dict[str, Any]] = None,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        self.tokenizer = tokenizer
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.system_prompt_length = len(
            initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        )

    async def _should_compress(self, state: ContextState) -> bool:
        session_state = state.extra_fields.get("session_state", {})
        return bool(session_state.get("pending_compression", False))

    async def _compress_impl(self, state: ContextState) -> ContextState:
        session_state = state.extra_fields.get("session_state", {})

        response_length = len(state.response_mask)
        if response_length == 0:
            logger.warning("compress called with empty response_mask, returning state unchanged")
            return state
        original_prompt_ids = state.trajectory_ids[:-response_length]

        round_summary_text = SearchAgentSession.render_handoff_from_state(session_state)

        summary_tokens = apply_chat_template(
            self.tokenizer,
            [{"role": "user", "content": round_summary_text}],
            add_generation_prompt=True,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        summary_ids = normalize_token_ids(summary_tokens)[self.system_prompt_length:]

        preserved_messages = []
        for message in state.messages:
            if message.get("role") in ("assistant", "tool"):
                break
            preserved_messages.append(dict(message))
        preserved_messages.append({"role": "user", "content": round_summary_text})

        return ContextState(
            messages=preserved_messages,
            trajectory_ids=list(original_prompt_ids) + list(summary_ids),
            response_mask=[0] * len(summary_ids),
            response_logprobs=[0.0] * len(summary_ids) if state.response_logprobs else [],
            multi_modal_data=dict(state.multi_modal_data),
            routed_experts=None,
            reward_score=state.reward_score,
            num_turns=len(preserved_messages),
            metrics=state.metrics.model_copy(deep=True),
            extra_fields=dict(state.extra_fields),
        )