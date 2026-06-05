# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SearchAgent tools — thin wrappers around SearchAgentSession.

search_corpus is a mock substring search over the config corpus (FAISS is a TODO).
Args are validated against each tool's own JSON Schema via jsonschema; mismatches
return {"invalid_format": [...]}. State/cross-field rules live in the session.
"""

import json
import logging
import traceback
from typing import Any, Optional

from jsonschema import Draft202012Validator

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .searchagent_session import SearchAgentSessionRegistry

logger = logging.getLogger(__name__)


def _error_response(message: Any) -> tuple[ToolResponse, float, dict]:
    return ToolResponse(text=json.dumps({"error": message}, ensure_ascii=False, default=str)), 0.0, {}


def _ok_response(payload: Any) -> tuple[ToolResponse, float, dict]:
    return ToolResponse(text=json.dumps(payload, ensure_ascii=False, default=str)), 0.0, {}


def _decode_args(parameters: Any) -> tuple[Any, Optional[str]]:
    """Normally already a dict (the loop json.loads()es arguments); decode a string layer too."""
    if isinstance(parameters, str):
        try:
            return json.loads(parameters), None
        except json.JSONDecodeError:
            return None, "arguments: not valid JSON"
    return parameters, None


def _schema_errors(instance: Any, params_schema: dict) -> list[str]:
    """JSON Schema violations as readable, path-anchored messages."""
    validator = Draft202012Validator(params_schema)
    errors: list[str] = []
    for e in sorted(validator.iter_errors(instance), key=lambda err: err.json_path):
        msg = f"{e.json_path}: {e.message}"
        errors.append(msg if len(msg) <= 200 else msg[:197] + "...")
    return errors


def _invalid_format_response(errors: list[str]) -> tuple[ToolResponse, float, dict]:
    return _error_response({"invalid_format": errors})


class SearchCorpusTool(BaseTool):
    """Semantic search over the document corpus. PoC: substring match over config corpus."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            _sd = {
                "type": "function",
                "function": {
                    "name": "search_corpus",
                    "description": (
                        "Find relevant chunks in the document corpus. Returns up to top_k "
                        "chunks, each with chunk_id, snippet text, and a relevance score. "
                        "Chunks already saved across rounds and chunks already returned "
                        "earlier in the current round are filtered out automatically — you "
                        "only see fresh content per call. Use multiple focused queries "
                        "within a round rather than one broad query. The session has a "
                        "limited total search budget; spend it carefully."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "minLength": 3,
                                "description": (
                                    "Focused natural-language query. Concrete entities and "
                                    "specific aspects work better than broad themes."
                                ),
                            },
                            "top_k": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": (
                                    "Maximum number of fresh chunks to return (default 5). "
                                    "Smaller values force more focused queries."
                                ),
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
            tool_schema = OpenAIFunctionToolSchema.model_validate(_sd)
            self._params_schema = _sd["function"]["parameters"]
        super().__init__(config, tool_schema)
        self.registry = SearchAgentSessionRegistry.get_instance()
        # TODO [out-of-poc]: replace with FAISS index loaded from config (path, embedder, etc.)
        self.mock_corpus: list[dict] = config.get("mock_corpus", [])

    async def execute(
        self, instance_id: str, parameters: Any, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        instance, derr = _decode_args(parameters)
        if derr is not None:
            return _invalid_format_response([derr])
        fmt_errors = _schema_errors(instance, self._params_schema)
        if fmt_errors:
            return _invalid_format_response(fmt_errors)

        request_id = kwargs.get("request_id", instance_id)
        session = self.registry.get_session(request_id)
        if session is None:
            return _error_response("session_not_found")

        query = instance["query"].strip()
        if not query:
            return _error_response("query_required")
        top_k = instance.get("top_k", 5)

        try:
            raw_results = self._mock_search(query, top_k)
            filtered, error = session.handle_search(query, raw_results)
        except Exception:
            logger.error(
                "search_corpus failed for request_id=%s\nparameters: %s\n%s",
                request_id,
                json.dumps(parameters, ensure_ascii=False, default=str)[:1000],
                traceback.format_exc(),
            )
            return _error_response("internal_error")

        if error is not None:
            return _error_response(error)

        payload = [
            {
                "chunk_id": r["chunk_id"],
                "snippet": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in filtered
        ]
        return _ok_response(payload)

    def _mock_search(self, query: str, top_k: int) -> list[dict]:
        # TODO [out-of-poc]: FAISS-backed semantic search. For now: token substring match.
        tokens = [t for t in query.lower().split() if t]
        if not tokens:
            return self.mock_corpus[:top_k]
        scored: list[tuple[int, dict]] = []
        for chunk in self.mock_corpus:
            text = chunk.get("content", "").lower()
            hits = sum(1 for t in tokens if t in text)
            if hits > 0:
                scored.append((hits, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


class SaveAndAdvanceTool(BaseTool):
    """Close current round: persist selected chunks, record summary, advance to next round."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            _sd = {
                "type": "function",
                "function": {
                    "name": "save_and_advance",
                    "description": (
                        "Close the current search round and open a new one with a fresh, "
                        "compressed context window. Chunks in saved_chunks survive across "
                        "rounds; everything else (raw search results, prior reasoning) is "
                        "discarded. Call this only after performing several focused "
                        "search_corpus calls in this round (typically 5 or more) that "
                        "returned promising results, and only if you want to keep "
                        "exploring further. Calling it too early (before enough searches) "
                        "is rejected. If you have already gathered enough material to "
                        "answer the query, call finalize_ranking instead. Never call this "
                        "with an empty saved_chunks list."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "saved_chunks": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {
                                            "type": "string",
                                            "description": (
                                                "ID of a chunk returned by an earlier "
                                                "search_corpus call. Must not be invented."
                                            ),
                                        },
                                        "reason": {
                                            "type": "string",
                                            "minLength": 20,
                                            "description": (
                                                "Concrete explanation of why this chunk is "
                                                "relevant to the user's query (>= 20 chars)."
                                            ),
                                        },
                                    },
                                    "required": ["chunk_id", "reason"],
                                },
                                "description": (
                                    "Non-empty array of OBJECTS, one per chunk to keep "
                                    "across rounds. Each item is {\"chunk_id\": <id from "
                                    "an earlier search_corpus result>, \"reason\": <text, "
                                    "at least 20 chars, why the chunk is relevant>}. "
                                    "Do not pass bare id strings."
                                ),
                            },
                            "drop_from_saved": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "chunk_ids previously saved that should now be "
                                    "removed (e.g. you found a better replacement). "
                                    "Must not overlap with saved_chunks."
                                ),
                                "default": [],
                            },
                            "covered_concepts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Atomic aspects of the original query you consider "
                                    "fully addressed by chunks gathered so far."
                                ),
                                "default": [],
                            },
                            "unresolved_concepts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Atomic aspects of the original query still open and "
                                    "to be addressed in the next round. Must not overlap "
                                    "with covered_concepts."
                                ),
                                "default": [],
                            },
                            "round_summary": {
                                "type": "string",
                                "minLength": 50,
                                "description": (
                                    "Plain-text recap of what was searched and what was "
                                    "found in the current round (>= 50 chars)."
                                ),
                            },
                            "next_round_goal": {
                                "type": "string",
                                "minLength": 30,
                                "description": (
                                    "Concrete plan for the next round: what to search "
                                    "for and which gaps to close (>= 30 chars)."
                                ),
                            },
                        },
                        "required": [
                            "saved_chunks",
                            "round_summary",
                            "next_round_goal",
                        ],
                    },
                },
            }
            tool_schema = OpenAIFunctionToolSchema.model_validate(_sd)
            self._params_schema = _sd["function"]["parameters"]
        super().__init__(config, tool_schema)
        self.registry = SearchAgentSessionRegistry.get_instance()

    async def execute(
        self, instance_id: str, parameters: Any, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        instance, derr = _decode_args(parameters)
        if derr is not None:
            return _invalid_format_response([derr])
        fmt_errors = _schema_errors(instance, self._params_schema)
        if fmt_errors:
            return _invalid_format_response(fmt_errors)

        request_id = kwargs.get("request_id", instance_id)
        session = self.registry.get_session(request_id)
        if session is None:
            return _error_response("session_not_found")

        try:
            errors = session.validate_save_and_advance(instance)
            if errors:
                return _error_response({"validation_failed": errors})
            session.commit_save_and_advance(instance)
        except Exception:
            logger.error(
                "save_and_advance failed for request_id=%s\nparameters: %s\n%s",
                request_id,
                json.dumps(parameters, ensure_ascii=False, default=str)[:1500],
                traceback.format_exc(),
            )
            return _error_response("internal_error")

        return _ok_response({
            "status": "advanced",
            "current_round": session.current_round,
            "saved_count": len(session.persistent_saved),
        })


class FinalizeRankingTool(BaseTool):
    """Terminate session and return the final ranked chunk list."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            _sd = {
                "type": "function",
                "function": {
                    "name": "finalize_ranking",
                    "description": (
                        "End the session and return the final ranked list of chunks. "
                        "Call this once you have gathered enough evidence to answer the "
                        "user's query. Each entry in ranking must reference a chunk_id "
                        "from a prior search_corpus result (or from chunks already saved "
                        "in earlier rounds). Order matters: most relevant chunk first. "
                        "Typical ranking size is 3–8 chunks; the maximum is 10."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ranking": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {
                                            "type": "string",
                                            "description": (
                                                "ID of a chunk returned by an earlier "
                                                "search_corpus call or saved in a prior "
                                                "round. Must not be invented."
                                            ),
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": (
                                                "Why this chunk earns its rank. Entries "
                                                "with reason shorter than 20 chars are "
                                                "dropped from the final ranking."
                                            ),
                                        },
                                    },
                                    "required": ["chunk_id", "reason"],
                                },
                                "description": (
                                    "Non-empty ranked array of OBJECTS, most relevant "
                                    "first. Each item is {\"chunk_id\": <id from a prior "
                                    "search result>, \"reason\": <text, at least 20 chars, "
                                    "why it earns this rank>}. Do not pass bare id strings."
                                ),
                            },
                            "step_evidence": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": (
                                    "Optional per-step evidence used only in plan-mode. "
                                    "Leave empty unless instructed otherwise."
                                ),
                                "default": [],
                            },
                        },
                        "required": ["ranking"],
                    },
                },
            }
            tool_schema = OpenAIFunctionToolSchema.model_validate(_sd)
            self._params_schema = _sd["function"]["parameters"]
        super().__init__(config, tool_schema)
        self.registry = SearchAgentSessionRegistry.get_instance()

    async def execute(
        self, instance_id: str, parameters: Any, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        instance, derr = _decode_args(parameters)
        if derr is not None:
            return _invalid_format_response([derr])
        fmt_errors = _schema_errors(instance, self._params_schema)
        if fmt_errors:
            return _invalid_format_response(fmt_errors)

        request_id = kwargs.get("request_id", instance_id)
        session = self.registry.get_session(request_id)
        if session is None:
            return _error_response("session_not_found")

        try:
            # Finalize is lenient: session filters bad ids / short reasons and caps to 10
            # (hence no minLength/maxItems in the schema), erroring only if nothing survives.
            errors = session.validate_finalize_ranking(instance)
            if errors:
                return _error_response({"validation_failed": errors})
            ranked = session.commit_finalize_ranking(instance)
        except Exception:
            logger.error(
                "finalize_ranking failed for request_id=%s\nparameters: %s\n%s",
                request_id,
                json.dumps(parameters, ensure_ascii=False, default=str)[:1500],
                traceback.format_exc(),
            )
            return _error_response("internal_error")

        payload = [
            {
                "rank": i + 1,
                "chunk_id": c.chunk_id,
                "snippet": c.content,
                "reason": c.reason,
                "score": c.score,
            }
            for i, c in enumerate(ranked)
        ]
        return _ok_response({"status": "finalized", "ranking": payload})