# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SearchAgent loop — generate → parse → execute → compress cycle.

One AgentLoopOutput is emitted per context-reset boundary (= per round in
SearchAgent terms). Terminates on successful finalize_ranking, exhausted
budgets, or session-level termination (e.g. consecutive locked turns).
"""

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.agent_loop_with_context_management import (
    AgentLoopWithContextManagement,
    ContextState,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import ToolResponse
from verl.tools.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

from .searchagent_context_manager import SearchAgentContextManager
from .searchagent_session import SearchAgentSession, SearchAgentSessionRegistry

logger = logging.getLogger(__name__)


@register("search_agent")
class SearchAgentLoop(AgentLoopWithContextManagement):
    """Multi-round search agent with structured-summary context compression."""

    def __init__(
        self,
        *args,
        max_rounds: int = 5,
        max_context_compressions: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_rounds = max_rounds
        self.max_context_compressions = max_context_compressions

        mt = self.rollout_config.multi_turn
        tool_list = initialize_tools_from_config(mt.tool_config_path) if mt.tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
        ]
        self.tool_parser_schemas = [tool.tool_schema for tool in tool_list]

        self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        self.tool_parser_name = mt.format
        self.max_parallel_calls = mt.max_parallel_calls
        self.max_tool_response_length = mt.max_tool_response_length
        self.tool_response_truncate_side = mt.tool_response_truncate_side
        self.max_assistant_turns = mt.max_assistant_turns
        self.max_user_turns = mt.max_user_turns

        self.context_manager = SearchAgentContextManager(
            tokenizer=self.tokenizer,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )
        self.registry = SearchAgentSessionRegistry.get_instance()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        messages = [dict(m) for m in list(kwargs["raw_prompt"])]

        if await self.process_vision_info(messages):
            raise ValueError("SearchAgentLoop only supports text prompts.")

        request_id = uuid4().hex
        session = self.registry.create_session(request_id=request_id, max_rounds=self.max_rounds)
        session.start_round()

        prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)
        state = ContextState(
            messages=messages,
            trajectory_ids=prompt_ids,
            num_turns=sum(1 for m in messages if m.get("role") != "system"),
            metrics=AgentLoopMetrics(),
            extra_fields={
                "request_id": request_id,
                "session_state": session.get_state_for_context(),
            },
        )

        outputs: list[AgentLoopOutput] = []
        compression_count = 0
        assistant_turns = 0
        tool_turns = 0

        try:
            while True:
                state, response_ids = await self._generate_next_state(
                    state=state,
                    request_id=request_id,
                    sampling_params=sampling_params,
                )
                assistant_turns += 1

                if self._should_terminate(state, assistant_turns, tool_turns):
                    outputs.append(self._build_output_from_state(state))
                    break

                _, tool_calls = await self.tool_parser.extract_tool_calls(
                    response_ids, self.tool_parser_schemas
                )
                if not tool_calls:
                    # Final text, no tool call → end of rollout.
                    # TODO [need decision]: TZ "free-text turns" (p.4) — maybe warn-and-continue,
                    #   force-terminate only on the 2nd consecutive one (with fallback ranking).
                    outputs.append(self._build_output_from_state(state))
                    break

                state, _ = await self._append_tool_responses(
                    state=state,
                    tool_calls=tool_calls,
                    tools_kwargs=kwargs.get("tools_kwargs", {}),
                    request_id=request_id,
                    session=session,
                )
                tool_turns += 1

                if any(tc.name == "finalize_ranking" for tc in tool_calls):
                    outputs.append(self._build_output_from_state(state))
                    break

                # TODO [need decision]: round-budget exhaustion just ends the segment;
                #   alternative is implicit save_and_advance / fallback so chunks aren't lost.
                if len(state.response_mask) >= self.response_length:
                    outputs.append(self._build_output_from_state(state))
                    break

                if compression_count < self.max_context_compressions:
                    next_state, compressed = await self.context_manager.check_and_compress(state)
                    if compressed:
                        outputs.append(self._build_output_from_state(state))
                        state = next_state
                        compression_count += 1
                        if not session.start_round():
                            # TODO [out-of-poc]: force-terminate should inject a fallback
                            #   ranking (persistent_saved + top-by-score) per TZ instead of
                            #   reward=0; same for the locked-out / free-text terminations.
                            break  # session terminated (max_rounds or locked-out)
                        # Refresh snapshot so the next iteration sees post-start_round state
                        # (e.g. pending_compression cleared, current_round incremented).
                        state.extra_fields["session_state"] = session.get_state_for_context()
        finally:
            self.registry.remove_session(request_id)

        return outputs

    async def _append_tool_responses(
        self,
        *,
        state: ContextState,
        tool_calls,
        tools_kwargs: dict[str, Any],
        request_id: str,
        session: SearchAgentSession,
    ) -> tuple[ContextState, list[float]]:
        metrics = state.metrics.model_dump()

        tasks = []
        tool_call_names: list[str] = []
        # TODO [out-of-poc]: reward penalty for exceeding max_parallel_calls (extra calls
        #   are currently sliced off silently; only meaningful once parallel calls > 1).
        for tc in tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tc, tools_kwargs, request_id))
            tool_call_names.append(tc.name)

        with simple_timer("tool_calls", metrics):
            responses = await asyncio.gather(*tasks)

        add_messages = [{"role": "tool", "content": (resp.text or "")} for resp, _ in responses]
        tool_rewards = [r for _, r in responses if r is not None]
        # TODO [out-of-poc]: tool_rewards don't reach the trainer — base _build_output_from_state
        #   overwrites extra_fields["tool_rewards"]=[]. Moot while reward is final-recall only.

        response_ids = await self._encode_tool_response_messages(add_messages, tool_call_names)

        extra_fields = dict(state.extra_fields)
        extra_fields["session_state"] = session.get_state_for_context()

        return (
            ContextState(
                messages=[dict(m) for m in state.messages] + add_messages,
                trajectory_ids=list(state.trajectory_ids) + response_ids,
                response_mask=list(state.response_mask) + [0] * len(response_ids),
                response_logprobs=(
                    list(state.response_logprobs) + [0.0] * len(response_ids)
                    if state.response_logprobs else []
                ),
                routed_experts=state.routed_experts,
                reward_score=state.reward_score,
                num_turns=state.num_turns + len(add_messages),
                metrics=AgentLoopMetrics(**metrics),
                extra_fields=extra_fields,
            ),
            tool_rewards,
        )

    async def _call_tool(
        self,
        tool_call,
        tools_kwargs: dict[str, Any],
        request_id: str,
    ) -> tuple[ToolResponse, float]:
        tool_name = tool_call.name
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResponse(text=json.dumps({"error": f"unknown_tool: {tool_name}"})), 0.0

        instance_id = None
        try:
            tool_args = json.loads(tool_call.arguments)
            per_tool = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=per_tool.get("create_kwargs", {}))
            response, reward, _ = await tool.execute(instance_id, tool_args, request_id=request_id)
        except Exception as e:
            logger.warning(f"[{request_id}] tool {tool_name} failed: {e}")
            return ToolResponse(text=json.dumps({"error": f"tool_exception: {e}"})), 0.0
        finally:
            if instance_id is not None:
                await tool.release(instance_id)

        text = response.text or ""
        if len(text) > self.max_tool_response_length:
            text = self._truncate(text)
        return ToolResponse(text=text), reward

    def _truncate(self, text: str) -> str:
        n = self.max_tool_response_length
        if self.tool_response_truncate_side == "left":
            return text[:n] + "...(truncated)"
        if self.tool_response_truncate_side == "right":
            return "(truncated)..." + text[-n:]
        half = n // 2
        return text[:half] + "...(truncated)..." + text[-half:]

    async def _encode_tool_response_messages(
        self,
        add_messages: list[dict[str, Any]],
        tool_call_names: list[str],
    ) -> list[int]:
        if self.tool_parser_name == "gpt-oss":
            text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            return await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(text, add_special_tokens=False)
            )
        return await self.apply_chat_template(add_messages, remove_system_prompt=True)

    def _should_terminate(self, state: ContextState, assistant_turns: int, tool_turns: int) -> bool:
        if len(state.response_mask) >= self.response_length:
            return True
        if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
            return True
        if self.max_user_turns and tool_turns >= self.max_user_turns:
            return True
        return False