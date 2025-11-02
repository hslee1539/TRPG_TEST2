"""Tests for the command line interface helpers in ``main``."""

from __future__ import annotations

import builtins
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest


class _StubChatOpenAI:  # pragma: no cover - used only for import compatibility
    def __init__(self, *args, **kwargs):
        raise RuntimeError("ChatOpenAI should be patched in tests")


class _StubLLMChain:  # pragma: no cover - used only for import compatibility
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def predict(self, *args, **kwargs):
        raise RuntimeError("LLMChain.predict should be patched in tests")


class _StubConversationBufferMemory:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubChatPromptTemplate:  # pragma: no cover
    @classmethod
    def from_messages(cls, messages):
        return (cls, messages)


class _StubMessagesPlaceholder:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubBaseMessage:
    def __init__(self, content: str):
        self.content = content


class _StubSystemMessage(_StubBaseMessage):
    pass


class _StubHumanMessage(_StubBaseMessage):
    pass


class _StubAIMessage(_StubBaseMessage):
    pass


def _install_langchain_stubs() -> None:
    langchain_module = types.ModuleType("langchain")

    chat_models_module = types.ModuleType("langchain.chat_models")
    chat_models_module.ChatOpenAI = _StubChatOpenAI

    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.ChatOpenAI = _StubChatOpenAI

    chains_module = types.ModuleType("langchain.chains")
    chains_module.LLMChain = _StubLLMChain

    memory_module = types.ModuleType("langchain.memory")
    memory_module.ConversationBufferMemory = _StubConversationBufferMemory

    prompts_module = types.ModuleType("langchain.prompts")
    prompts_module.ChatPromptTemplate = _StubChatPromptTemplate
    prompts_module.MessagesPlaceholder = _StubMessagesPlaceholder

    schema_module = types.ModuleType("langchain.schema")
    schema_module.BaseMessage = _StubBaseMessage
    schema_module.SystemMessage = _StubSystemMessage
    schema_module.HumanMessage = _StubHumanMessage
    schema_module.AIMessage = _StubAIMessage

    sys.modules.setdefault("langchain", langchain_module)
    sys.modules.setdefault("langchain.chat_models", chat_models_module)
    sys.modules.setdefault("langchain_openai", langchain_openai_module)
    sys.modules.setdefault("langchain.chains", chains_module)
    sys.modules.setdefault("langchain.memory", memory_module)
    sys.modules.setdefault("langchain.prompts", prompts_module)
    sys.modules.setdefault("langchain.schema", schema_module)


_install_langchain_stubs()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402  (import after path manipulation)


class _DummyGameMaster:
    """Lightweight fake ``GameMaster`` used to exercise ``prompt_loop``."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.inputs: list[str] = []
        self.scene = "(initial scene)"

    def respond(self, player_input: str) -> str:
        self.inputs.append(player_input)
        if not self._responses:
            raise AssertionError("No responses left in dummy game master")
        response = self._responses.pop(0)
        self.scene = f"Scene: {player_input} -> {response}"
        return response

    def render_scene(self, width: int = 60) -> str:
        return self.scene


def test_build_llm_uses_explicit_arguments() -> None:
    """``build_llm`` should pass through provided API details to ``ChatOpenAI``."""

    mocked_llm = mock.MagicMock(name="ChatOpenAI return")

    with mock.patch.object(main, "ChatOpenAI", return_value=mocked_llm) as ctor:
        result = main.build_llm(
            model="custom-model",
            temperature=0.42,
            api_base="http://example.test",
            api_key="secret-key",
        )

    assert result is mocked_llm
    ctor.assert_called_once_with(
        model="custom-model",
        temperature=0.42,
        openai_api_base="http://example.test",
        openai_api_key="secret-key",
    )


def test_build_llm_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should be consulted when arguments are omitted."""

    monkeypatch.setenv("LM_STUDIO_API_BASE", "http://env-base")
    monkeypatch.setenv("LM_STUDIO_API_KEY", "env-key")

    mocked_llm = mock.MagicMock(name="ChatOpenAI return")

    with mock.patch.object(main, "ChatOpenAI", return_value=mocked_llm) as ctor:
        result = main.build_llm(model="m", temperature=0.1)

    assert result is mocked_llm
    ctor.assert_called_once_with(
        model="m",
        temperature=0.1,
        openai_api_base="http://env-base",
        openai_api_key="env-key",
    )


def test_build_game_master_uses_helper() -> None:
    """``build_game_master`` should use ``create_default_game_master``."""

    fake_game_master = SimpleNamespace()

    with mock.patch.object(main, "build_llm", return_value="llm") as build_llm:
        with mock.patch.object(
            main, "create_default_game_master", return_value=fake_game_master
        ) as factory:
            result = main.build_game_master("model", 0.5, api_base="base", api_key="key")

    build_llm.assert_called_once_with(
        model="model", temperature=0.5, api_base="base", api_key="key"
    )
    factory.assert_called_once_with("llm")
    assert result is fake_game_master


def test_prompt_loop_handles_quit(capsys: pytest.CaptureFixture[str]) -> None:
    """The loop should stop when the player types ``quit``."""

    gm = _DummyGameMaster(["Response 1"])

    inputs = iter(["Hello", "quit"])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    with mock.patch.object(builtins, "input", side_effect=fake_input):
        main.prompt_loop(gm, initial_prompt="Intro")

    captured = capsys.readouterr()
    assert "Intro" in captured.out
    assert "Response 1" in captured.out
    assert "Scene: Hello -> Response 1" in captured.out
    # The dummy GM should have received only the non-quit input.
    assert gm.inputs == ["Hello"]


def test_main_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` should build the game master and enter the prompt loop."""

    fake_gm = object()

    with mock.patch.object(main, "parse_args", return_value=SimpleNamespace(
        model="model",
        temperature=0.9,
        api_base=None,
        api_key=None,
        input_mode="text",
    )) as parse_args:
        with mock.patch.object(main, "build_game_master", return_value=fake_gm) as build:
            with mock.patch.object(main, "prompt_loop") as loop:
                exit_code = main.main(["--anything"])

    assert exit_code == 0
    parse_args.assert_called_once_with(["--anything"])
    build.assert_called_once_with(model="model", temperature=0.9, api_base=None, api_key=None)
    loop.assert_called_once()
    assert loop.call_args.kwargs["input_mode"] == "text"


def test_main_handles_initialization_errors() -> None:
    """``main`` should emit errors and return a failing exit code on exceptions."""

    with mock.patch.object(main, "parse_args", return_value=SimpleNamespace(
        model="m",
        temperature=0.1,
        api_base=None,
        api_key=None,
        input_mode="text",
    )):
        with mock.patch.object(main, "build_game_master", side_effect=RuntimeError("boom")):
            with mock.patch.object(sys, "stderr") as fake_stderr:
                exit_code = main.main()

    assert exit_code == 1
    fake_stderr.write.assert_called()  # Ensure an error was surfaced to the user.


def test_prompt_loop_voice_initialization_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Voice mode should exit gracefully if initialization fails."""

    gm = _DummyGameMaster([])

    with mock.patch.object(main, "_initialize_voice_capture", side_effect=RuntimeError("no mic")):
        main.prompt_loop(gm, input_mode="voice")

    captured = capsys.readouterr()
    assert "no mic" in captured.err


def test_prompt_loop_voice_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop should process recognized speech and stop when the player says quit."""

    gm = _DummyGameMaster(["Voice response"])

    voice_inputs = iter([None, "Hello", "quit"])

    monkeypatch.setattr(main, "_initialize_voice_capture", lambda: ("recognizer", "microphone"))

    def fake_capture(recognizer, microphone):
        return next(voice_inputs)

    monkeypatch.setattr(main, "_capture_voice_input", fake_capture)

    main.prompt_loop(gm, input_mode="voice")

    assert gm.inputs == ["Hello"]


def test_capture_voice_input_uses_korean_language() -> None:
    """Speech recognition should request transcription in Korean."""

    recognizer = mock.MagicMock()
    microphone = mock.MagicMock()
    microphone.__enter__.return_value = microphone
    audio = mock.sentinel.audio

    recognizer.listen.return_value = audio
    recognizer.recognize_google.return_value = "안녕하세요"

    result = main._capture_voice_input(recognizer, microphone)

    assert result == "안녕하세요"
    recognizer.listen.assert_called_once_with(microphone)
    recognizer.recognize_google.assert_called_once_with(
        audio, language=main.DEFAULT_SPEECH_LANGUAGE
    )
