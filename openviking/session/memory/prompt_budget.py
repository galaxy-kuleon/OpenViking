# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Bound archived-conversation prompt text without losing the newest intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from openviking.utils.token_estimation import estimate_text_tokens

_MARKER_RESERVE_TOKENS = 64
_MIN_BOUNDARY_SECTION_TOKENS = 64
_TRUNCATION_MARKER = "\n[... section truncated for prompt budget ...]\n"


@dataclass(frozen=True)
class PromptBudgetResult:
    text: str
    original_tokens: int
    final_tokens: int
    included_sections: int
    omitted_sections: int
    truncated_sections: int


def _truncate_text(text: str, max_tokens: int) -> str:
    """Retain both ends of one section inside a conservative token budget."""
    if estimate_text_tokens(text) <= max_tokens:
        return text
    marker_tokens = estimate_text_tokens(_TRUNCATION_MARKER)
    if max_tokens <= marker_tokens:
        return _TRUNCATION_MARKER.strip()

    low = 0
    high = len(text)
    best = _TRUNCATION_MARKER.strip()
    while low <= high:
        retained_chars = (low + high) // 2
        head_chars = retained_chars // 3
        tail_chars = retained_chars - head_chars
        candidate = (
            text[:head_chars] + _TRUNCATION_MARKER + (text[-tail_chars:] if tail_chars else "")
        )
        if estimate_text_tokens(candidate) <= max_tokens:
            best = candidate
            low = retained_chars + 1
        else:
            high = retained_chars - 1
    return best


def budget_prompt_sections(
    sections: Iterable[str],
    *,
    max_tokens: int,
) -> PromptBudgetResult:
    """Keep a contiguous recent suffix and explicitly mark any degradation."""
    normalized = [str(section) for section in sections if str(section)]
    full_text = "\n".join(normalized)
    original_tokens = estimate_text_tokens(full_text)
    if original_tokens <= max_tokens:
        return PromptBudgetResult(
            text=full_text,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            included_sections=len(normalized),
            omitted_sections=0,
            truncated_sections=0,
        )

    available = max(0, max_tokens - _MARKER_RESERVE_TOKENS)
    selected: list[str] = []
    selected_tokens = 0
    boundary_index = len(normalized)
    truncated_sections = 0

    for index in range(len(normalized) - 1, -1, -1):
        section = normalized[index]
        separator_tokens = estimate_text_tokens("\n") if selected else 0
        section_tokens = estimate_text_tokens(section)
        if selected_tokens + separator_tokens + section_tokens <= available:
            selected.insert(0, section)
            selected_tokens += separator_tokens + section_tokens
            boundary_index = index
            continue

        remaining = available - selected_tokens - separator_tokens
        if remaining >= _MIN_BOUNDARY_SECTION_TOKENS or not selected:
            selected.insert(0, _truncate_text(section, max(remaining, 1)))
            boundary_index = index
            truncated_sections = 1
        break

    omitted_sections = boundary_index
    marker = (
        "[Prompt budget applied: "
        f"{omitted_sections} earlier section(s) omitted; "
        f"{truncated_sections} boundary section(s) truncated.]"
    )
    final_text = "\n".join([marker, *selected])
    final_tokens = estimate_text_tokens(final_text)
    if final_tokens > max_tokens and selected:
        selected[0] = _truncate_text(
            selected[0],
            max(1, estimate_text_tokens(selected[0]) - (final_tokens - max_tokens)),
        )
        truncated_sections = 1
        final_text = "\n".join([marker, *selected])
        final_tokens = estimate_text_tokens(final_text)

    return PromptBudgetResult(
        text=final_text,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        included_sections=len(selected),
        omitted_sections=omitted_sections,
        truncated_sections=truncated_sections,
    )
