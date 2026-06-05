# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SearchAgent session and registry.

PoC scope: structural skeleton with hooks for all TZ rules. Most validation
is stubbed with TODOs — only the minimal subset needed for the loop to flow
end-to-end is implemented. See _todo_business_rules for the full list.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SavedChunk:
    chunk_id: str
    content: str
    reason: str = ""
    score: float = 0.0


@dataclass
class RoundSummary:
    round_number: int
    queries: list[str] = field(default_factory=list)
    saved_chunk_ids: list[str] = field(default_factory=list)
    round_summary: str = ""
    next_round_goal: str = ""


class SearchAgentSession:
    """
    Per-rollout state and validation logic for SearchAgent.

    Two-phase tool integration:
        validate_*(params) -> list[str]      # pure check, no mutation
        commit_*(params)   -> result         # apply state changes

    Tools call validate first, return errors in observation if non-empty,
    otherwise call commit.
    """

    def __init__(
        self,
        request_id: str,
        max_rounds: int = 5,
        search_budget: int = 20,
        hard_lock_threshold: float = 0.75,
        max_consecutive_locked_turns: int = 2,
        min_searches_before_save: int = 5,
        min_reason_length: int = 20,
        min_round_summary_length: int = 50,
        min_next_round_goal_length: int = 30,
        max_ranking_size: int = 10,
    ):
        self.request_id = request_id
        self.max_rounds = max_rounds
        self.search_budget = search_budget
        self.hard_lock_threshold = hard_lock_threshold
        self.max_consecutive_locked_turns = max_consecutive_locked_turns
        self.min_searches_before_save = min_searches_before_save
        self.min_reason_length = min_reason_length
        self.min_round_summary_length = min_round_summary_length
        self.min_next_round_goal_length = min_next_round_goal_length
        self.max_ranking_size = max_ranking_size

        # Persistent state — survives context resets
        self.persistent_saved: dict[str, SavedChunk] = {}
        self.round_summaries: list[RoundSummary] = []
        self.covered_concepts: set[str] = set()
        self.unresolved_concepts: set[str] = set()
        self.global_seen_chunk_ids: set[str] = set()
        self.search_count: int = 0
        self._query_history_normalized: set[str] = set()

        self._chunks_seen_storage: dict[str, dict[str, Any]] = {}

        # Round-local state — reset on advance
        self._current_round: int = 0
        self.seen_this_round: set[str] = set()
        self.queries_this_round: list[str] = []
        self.consecutive_locked_turns: int = 0
        self.save_and_advance_called_this_round: bool = False

        self.finalize_called: bool = False

        self._pending_compression: bool = False

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def is_last_round(self) -> bool:
        return self._current_round >= self.max_rounds

    @property
    def is_search_locked(self) -> bool:
        return self.search_count >= self.hard_lock_threshold * self.search_budget

    @property
    def is_terminated(self) -> bool:
        return (
            self.finalize_called
            or self.consecutive_locked_turns >= self.max_consecutive_locked_turns
        )

    def start_round(self) -> bool:
        """Begin a new round. Returns False if session is terminated or max_rounds reached."""
        if self.is_terminated or self._current_round >= self.max_rounds:
            return False
        self._current_round += 1
        self.seen_this_round = set()
        self.queries_this_round = []
        self.save_and_advance_called_this_round = False
        self._pending_compression = False
        return True

    def handle_search(
        self, query: str, raw_results: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """
        Process search results from the corpus: enforce filters and protective rules.

        Args:
            query: the search query string (already stripped/normalized by the tool).
            raw_results: results from the tool's corpus call. Each item must have
                at least `chunk_id` and `content`; `score` is optional.

        Returns:
            (filtered_results, error). On error, filtered_results is [].
        """
        if self.is_search_locked:
            self.consecutive_locked_turns += 1
            return [], "search_locked"

        # Duplicate query rejection. Budget is NOT charged and the turn is "lost"
        normalized_query = " ".join(query.lower().split())
        if normalized_query in self._query_history_normalized:
            return [], "duplicate_query"
        self._query_history_normalized.add(normalized_query)

        # TODO [out-of-poc]: anti-repeat hash for near-duplicate queries (second-tier
        #   spam guard on top of exact duplicate_query; not needed on the toy corpus).

        self.search_count += 1
        self.queries_this_round.append(query)
        self.consecutive_locked_turns = 0

        filtered: list[dict[str, Any]] = []
        for result in raw_results:
            chunk_id = result.get("chunk_id")
            if chunk_id is None:
                continue
            self._chunks_seen_storage[chunk_id] = result
            if chunk_id in self.persistent_saved:
                continue  # already saved, not surfaced again
            if chunk_id in self.seen_this_round:
                continue  # intra-round dedup
            filtered.append(result)
            self.seen_this_round.add(chunk_id)
            self.global_seen_chunk_ids.add(chunk_id)

        return filtered, None

    def validate_save_and_advance(self, params: dict[str, Any]) -> list[str]:
        """Check save_and_advance parameters against TZ rules. Returns error messages or []."""
        errors: list[str] = []

        if self.save_and_advance_called_this_round:
            errors.append("save_and_advance already called this round")
        if self.is_last_round:
            errors.append("save_and_advance not allowed in the last round, use finalize_ranking")

        saved_chunks = params.get("saved_chunks", [])
        if not isinstance(saved_chunks, list) or len(saved_chunks) == 0:
            errors.append("saved_chunks must be a non-empty list")

        errors.extend(self._check_save_and_advance_rules(params))
        return errors

    def _check_save_and_advance_rules(self, params: dict[str, Any]) -> list[str]:
        """Business rules a static schema can't express: runtime membership and
        cross-field disjointness. Structural/length constraints live in the schema."""
        errors: list[str] = []

        saved_chunks = params.get("saved_chunks", []) or []
        drop_from_saved = set(params.get("drop_from_saved", []) or [])
        covered = set(params.get("covered_concepts", []) or [])
        unresolved = set(params.get("unresolved_concepts", []) or [])

        # (1) all saved chunk_ids must come from seen_this_round ∪ persistent_saved
        allowed_ids = self.seen_this_round | set(self.persistent_saved.keys())
        unknown = [
            e.get("chunk_id") for e in saved_chunks
            if e.get("chunk_id") and e.get("chunk_id") not in allowed_ids
        ]
        if unknown:
            errors.append(f"unknown chunk_ids (not seen this round or saved): {unknown}")

        # (2) saved_chunks and drop_from_saved must not intersect
        saved_ids = {e.get("chunk_id") for e in saved_chunks if e.get("chunk_id")}
        overlap_saved_drop = saved_ids & drop_from_saved
        if overlap_saved_drop:
            errors.append(f"saved_chunks and drop_from_saved overlap: {sorted(overlap_saved_drop)}")

        # (3) covered_concepts and unresolved_concepts must not intersect
        overlap_concepts = covered & unresolved
        if overlap_concepts:
            errors.append(f"covered and unresolved concepts overlap: {sorted(overlap_concepts)}")

        # TODO [need decision]: >= min_searches_before_save *successful* searches before
        #   save. Needs def of "successful" + whether smoke's 5 should be relaxed (top-#5).
        # TODO [need decision]: require >=1 of covered/unresolved non-empty. TZ wants it,
        #   but both are schema-optional — enforcing makes them de-facto required.

        return errors

    def commit_save_and_advance(self, params: dict[str, Any]) -> None:
        """Apply save_and_advance mutations. Caller must validate first."""
        for chunk_id in params.get("drop_from_saved", []):
            self.persistent_saved.pop(chunk_id, None)

        saved_chunks = params.get("saved_chunks", [])
        for entry in saved_chunks:
            chunk_id = entry["chunk_id"]  # schema guarantees presence
            stored = self._chunks_seen_storage.get(chunk_id, {})
            self.persistent_saved[chunk_id] = SavedChunk(
                chunk_id=chunk_id,
                content=stored.get("content", ""),
                reason=entry.get("reason", ""),
                score=float(stored.get("score", 0.0)),
            )

        # TODO [need decision]: concepts union (current) vs replace? Union means a concept
        #   once covered can't return to unresolved (we subtract covered from unresolved).
        self.covered_concepts.update(params.get("covered_concepts", []))
        self.unresolved_concepts.update(params.get("unresolved_concepts", []))
        self.unresolved_concepts -= self.covered_concepts

        self.round_summaries.append(
            RoundSummary(
                round_number=self._current_round,
                queries=list(self.queries_this_round),
                saved_chunk_ids=[e["chunk_id"] for e in saved_chunks],
                round_summary=params.get("round_summary", ""),
                next_round_goal=params.get("next_round_goal", ""),
            )
        )

        # Discard seen-but-not-saved chunks
        self._chunks_seen_storage = {
            cid: data for cid, data in self._chunks_seen_storage.items()
            if cid in self.persistent_saved
        }

        self.save_and_advance_called_this_round = True
        self.consecutive_locked_turns = 0
        self._pending_compression = True

    def validate_finalize_ranking(self, params: dict[str, Any]) -> list[str]:
        """Check finalize_ranking parameters against TZ rules. Returns error messages or []."""
        errors: list[str] = []

        if self.finalize_called:
            errors.append("finalize_ranking already called")

        ranking = params.get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) == 0:
            errors.append("ranking must be a non-empty list")
            return errors  # nothing to filter

        # Invalid chunk_ids and short reasons are *filtered*, not rejected — the
        # call only fails if NOTHING survives filtering.
        kept, dropped_invalid, dropped_short = self._filter_ranking(ranking)
        if not kept:
            errors.append(
                "ranking is empty after filtering "
                f"(invalid chunk_ids: {dropped_invalid}, short reasons: {dropped_short})"
            )
        return errors

    def _filter_ranking(
        self, ranking: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """
        Pure filter shared by validate and commit (no mutation, order preserved).

        Returns (kept_entries, dropped_invalid_chunk_ids, dropped_short_reason_ids):
        - drop entries whose chunk_id ∉ persistent_saved ∪ seen_this_round
        - drop entries whose reason is shorter than min_reason_length
        Cap to max_ranking_size is applied by the caller (commit), not here.
        """
        allowed_ids = self.seen_this_round | set(self.persistent_saved.keys())
        kept: list[dict[str, Any]] = []
        dropped_invalid: list[str] = []
        dropped_short: list[str] = []
        for entry in ranking:
            if not isinstance(entry, dict):
                continue
            chunk_id = entry.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id not in allowed_ids:
                dropped_invalid.append(chunk_id)
                continue
            if len(str(entry.get("reason", ""))) < self.min_reason_length:
                dropped_short.append(chunk_id)
                continue
            kept.append(entry)
        return kept, dropped_invalid, dropped_short

    def commit_finalize_ranking(self, params: dict[str, Any]) -> list[SavedChunk]:
        """Apply finalize_ranking and return the final ranked list. Caller must validate first."""
        ranking = params.get("ranking", [])

        # Re-run the same pure filter validate used, then cap. Membership and
        # reason-length are already enforced here, so entries are guaranteed valid.
        kept, dropped_invalid, dropped_short = self._filter_ranking(ranking)
        if dropped_invalid or dropped_short:
            logger.info(
                "[%s] finalize_ranking dropped invalid=%s short_reason=%s",
                self.request_id, dropped_invalid, dropped_short,
            )
        kept = kept[: self.max_ranking_size]

        result: list[SavedChunk] = []
        for entry in kept:
            chunk_id = entry["chunk_id"]
            # chunk_id ∈ persistent_saved ∪ seen_this_round (guaranteed by _filter_ranking).
            # Content comes from persistent_saved if saved, else from this round's storage.
            base = self.persistent_saved.get(chunk_id)
            if base is None:
                stored = self._chunks_seen_storage.get(chunk_id, {})
                base = SavedChunk(
                    chunk_id=chunk_id,
                    content=stored.get("content", ""),
                    reason="",
                    score=float(stored.get("score", 0.0)),
                )
            result.append(SavedChunk(
                chunk_id=base.chunk_id,
                content=base.content,
                reason=entry.get("reason", base.reason),
                score=base.score,
            ))

        self.finalize_called = True
        return result

    def get_state_for_context(self) -> dict[str, Any]:
        """Snapshot used by ContextManager when rebuilding context after compression."""
        return {
            "current_round": self._current_round,
            "search_count": self.search_count,
            "pending_compression": self._pending_compression,
            "persistent_saved": [
                {"chunk_id": c.chunk_id, "content": c.content, "reason": c.reason, "score": c.score}
                for c in self.persistent_saved.values()
            ],
            "round_summaries": [
                {
                    "round_number": s.round_number,
                    "queries": s.queries,
                    "saved_chunk_ids": s.saved_chunk_ids,
                    "round_summary": s.round_summary,
                    "next_round_goal": s.next_round_goal,
                }
                for s in self.round_summaries
            ],
            "covered_concepts": sorted(self.covered_concepts),
            "unresolved_concepts": sorted(self.unresolved_concepts),
        }

    @staticmethod
    def render_handoff_from_state(session_state: dict[str, Any]) -> str:
        """Cross-round handoff text injected atop a fresh round. Pure function of the
        get_state_for_context() snapshot, so the ContextManager stays thin.
        NOTE: the "# Context from prior rounds" first line is load-bearing for the
        rollout viewer's compressed-block detection — keep it first."""
        payload = {
            "saved_chunks_available_for_finalize": [
                {"chunk_id": c["chunk_id"], "fact": (c.get("content") or "")[:160]}
                for c in session_state.get("persistent_saved", [])
            ],
            "prior_rounds": [
                {
                    "round": r["round_number"],
                    "summary": r.get("round_summary", ""),
                    "next_goal": r.get("next_round_goal", ""),
                }
                for r in session_state.get("round_summaries", [])
            ],
            "covered_concepts": session_state.get("covered_concepts", []),
            "unresolved_concepts": session_state.get("unresolved_concepts", []),
        }

        body = yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True, default_flow_style=False, width=88
        )
        next_round = session_state.get("current_round", 0) + 1

        return (
            "# Context from prior rounds\n"
            "\n"
            f"```yaml\n{body.rstrip()}\n```\n"
            "\n"
            f"Round {next_round} starts now. Either continue searching with search_corpus, "
            "save_and_advance again, or finalize_ranking using the saved chunks above."
        )


class SearchAgentSessionRegistry:
    """Singleton registry sharing sessions between SearchAgentLoop and tools."""

    _instance: Optional["SearchAgentSessionRegistry"] = None
    _sessions: dict[str, SearchAgentSession]

    def __new__(cls) -> "SearchAgentSessionRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SearchAgentSessionRegistry":
        return cls()

    def create_session(self, request_id: str, **kwargs) -> SearchAgentSession:
        if request_id in self._sessions:
            logger.warning(f"Session {request_id} already exists, reusing")
            return self._sessions[request_id]
        session = SearchAgentSession(request_id=request_id, **kwargs)
        self._sessions[request_id] = session
        return session

    def get_session(self, request_id: str) -> Optional[SearchAgentSession]:
        return self._sessions.get(request_id)

    def remove_session(self, request_id: str) -> None:
        self._sessions.pop(request_id, None)

    def clear_all(self) -> None:
        self._sessions.clear()