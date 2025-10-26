"""Tests for the ``GameMaster`` orchestration helpers."""

from __future__ import annotations

import pathlib
import sys
import types

import pytest


class _StubBaseMessage:
    def __init__(self, content: str):
        self.content = content


class _StubSystemMessage(_StubBaseMessage):
    pass


class _StubHumanMessage(_StubBaseMessage):
    pass


class _StubAIMessage(_StubBaseMessage):
    pass


def _install_message_stubs() -> None:
    core_module = types.ModuleType("langchain_core")
    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.BaseMessage = _StubBaseMessage
    messages_module.SystemMessage = _StubSystemMessage
    messages_module.HumanMessage = _StubHumanMessage
    messages_module.AIMessage = _StubAIMessage
    core_module.messages = messages_module

    schema_module = types.ModuleType("langchain.schema")
    schema_module.BaseMessage = _StubBaseMessage
    schema_module.SystemMessage = _StubSystemMessage
    schema_module.HumanMessage = _StubHumanMessage
    schema_module.AIMessage = _StubAIMessage

    sys.modules.setdefault("langchain_core", core_module)
    sys.modules.setdefault("langchain_core.messages", messages_module)
    sys.modules.setdefault("langchain.schema", schema_module)


_install_message_stubs()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trpg.game_master import AIMessage, GameMaster, SystemMessage, create_default_game_master


class _DummyInvokeLLM:
    """LLM stub that exposes the new ``invoke`` API."""

    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.response)


def test_game_master_updates_state_and_history() -> None:
    """``respond`` should call the LLM and track conversation history."""

    llm = _DummyInvokeLLM("The GM continues the tale.")
    gm = create_default_game_master(llm)

    reply = gm.respond("Hello there")

    assert reply == "The GM continues the tale."
    assert len(llm.calls) == 1
    sent_messages = llm.calls[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[-1].content.startswith("Facts so far:\n")
    assert gm.state.facts[-2:] == ["Player: Hello there", "GM: The GM continues the tale."]


class _DummyPredictMessagesLLM:
    """LLM stub that only offers the legacy ``predict_messages`` API."""

    def __init__(self):
        self.calls = []

    def predict_messages(self, messages):  # pragma: no cover - legacy compatibility
        self.calls.append(messages)
        return AIMessage(content="Legacy path")


def test_game_master_supports_legacy_predict_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older LangChain versions expose ``predict_messages`` instead of ``invoke``."""

    llm = _DummyPredictMessagesLLM()
    gm = GameMaster(llm, system_template="System", state=None, initial_facts=None)

    reply = gm.respond("Hi")

    assert reply == "Legacy path"
    assert llm.calls, "The legacy LLM should have been invoked"


class _DummyCallableLLM:
    """LLM stub that behaves like a bare callable returning raw strings."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages):  # pragma: no cover - highly defensive
        self.calls.append(messages)
        return "Callable path"


def test_game_master_accepts_callable_llms() -> None:
    """As a last resort the GM should treat the LLM as a callable."""

    llm = _DummyCallableLLM()
    gm = GameMaster(llm, system_template="System", state=None, initial_facts=None)

    reply = gm.respond("Greetings")

    assert reply == "Callable path"
    assert llm.calls, "Callable LLM should have been invoked"
