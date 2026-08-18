from openviking.message import Message
from openviking.message.part import TextPart
from openviking.session.memory.prompt_budget import budget_prompt_sections
from openviking.session.memory.session_extract_context_provider import (
    SessionExtractContextProvider,
)
from openviking.session.session import Session


def test_prompt_budget_leaves_small_input_unchanged():
    result = budget_prompt_sections(["first", "second"], max_tokens=512)

    assert result.text == "first\nsecond"
    assert result.omitted_sections == 0
    assert result.truncated_sections == 0


def test_prompt_budget_keeps_recent_suffix_and_marks_omission():
    result = budget_prompt_sections(
        ["old " * 300, "middle " * 300, "newest intent"],
        max_tokens=128,
    )

    assert "newest intent" in result.text
    assert "Prompt budget applied" in result.text
    assert "old " * 20 not in result.text
    assert result.final_tokens <= 128
    assert result.omitted_sections > 0 or result.truncated_sections == 1


def test_prompt_budget_retains_both_ends_of_oversized_latest_section():
    result = budget_prompt_sections(
        ["BEGIN " + ("中" * 500) + " END"],
        max_tokens=128,
    )

    assert "BEGIN" in result.text
    assert "END" in result.text
    assert "section truncated" in result.text
    assert result.final_tokens <= 128


def test_session_provider_preserves_original_indices_after_budgeting(monkeypatch):
    class MemoryConfig:
        eager_prefetch = False
        prefetch_search_topn = 5
        extraction_prompt_max_tokens = 2048
        link_enabled = False

    class Config:
        memory = MemoryConfig()

    monkeypatch.setattr(
        "openviking.session.memory.session_extract_context_provider.get_openviking_config",
        lambda: Config(),
    )
    monkeypatch.setattr(SessionExtractContextProvider, "_detect_language", lambda self: "en")
    messages = [
        Message(id=f"m{index}", role="user", parts=[TextPart(text)])
        for index, text in enumerate(["old " * 300, "middle " * 300, "latest request"])
    ]
    provider = SessionExtractContextProvider(messages=messages)
    provider._extraction_prompt_max_tokens = 128

    conversation = provider._assemble_conversation(messages)

    assert "[2][user][user]: latest request" in conversation
    assert "[0][user][user]" not in conversation
    assert "Prompt budget applied" in conversation


def test_working_memory_prompt_uses_the_same_recent_message_budget(monkeypatch):
    class MemoryConfig:
        extraction_prompt_max_tokens = 128

    class Config:
        memory = MemoryConfig()

    monkeypatch.setattr(
        "openviking.session.session.get_openviking_config",
        lambda: Config(),
    )
    messages = [
        Message(id="old", role="user", parts=[TextPart("old " * 500)]),
        Message(id="new", role="user", parts=[TextPart("latest request")]),
    ]

    formatted = Session._format_messages_for_wm(messages)

    assert "latest request" in formatted
    assert "old " * 20 not in formatted
    assert "Prompt budget applied" in formatted
