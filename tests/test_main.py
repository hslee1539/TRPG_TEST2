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


def test_prepare_speech_segments_splits_sentences() -> None:
    """Speech text should be broken into short, punctuated segments."""

    text = "첫 문장입니다! 두 번째 문장\n세 번째 문장"

    segments = main._prepare_speech_segments(text)

    assert segments == [
        "첫 문장입니다!",
        "두 번째 문장.",
        "세 번째 문장.",
    ]


def test_speak_text_uses_prepared_segments() -> None:
    """The TTS helper should feed each prepared segment to the engine."""

    engine = mock.Mock()

    with mock.patch.object(main, "_prepare_speech_segments", return_value=["하나.", "둘."]):
        main._speak_text(engine, "ignored")

    engine.say.assert_has_calls([mock.call("하나."), mock.call("둘.")])
    engine.runAndWait.assert_called_once_with()


def test_main_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` should build the game master and enter the prompt loop."""

    fake_gm = object()

    with mock.patch.object(main, "parse_args", return_value=SimpleNamespace(
        model="model",
        temperature=0.9,
        api_base=None,
        api_key=None,
        input_mode="text",
        speak_gm=False,
    )) as parse_args:
        with mock.patch.object(main, "build_game_master", return_value=fake_gm) as build:
            with mock.patch.object(main, "prompt_loop") as loop:
                exit_code = main.main(["--anything"])

    assert exit_code == 0
    parse_args.assert_called_once_with(["--anything"])
    build.assert_called_once_with(model="model", temperature=0.9, api_base=None, api_key=None)
    loop.assert_called_once()
    assert loop.call_args.kwargs["input_mode"] == "text"
    assert loop.call_args.kwargs["speak_gm"] is False


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


def test_prompt_loop_voice_output_initialization_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Voice output errors should be surfaced but not crash the loop."""

    gm = _DummyGameMaster(["Ignored"])

    with mock.patch.object(main, "_initialize_voice_output", side_effect=RuntimeError("no tts")):
        with mock.patch.object(builtins, "input", side_effect=["quit"]):
            with mock.patch.object(main, "_speak_text") as speak:
                main.prompt_loop(gm, speak_gm=True)

    speak.assert_not_called()
    captured = capsys.readouterr()
    assert "no tts" in captured.err


def test_prompt_loop_speaks_responses() -> None:
    """When enabled, the GM's replies should be spoken via the TTS helper."""

    gm = _DummyGameMaster(["First response"])

    inputs = iter(["Hello", "quit"])

    with mock.patch.object(main, "_initialize_voice_output", return_value="engine") as init:
        with mock.patch.object(builtins, "input", side_effect=lambda _: next(inputs)):
            with mock.patch.object(main, "_speak_text") as speak:
                main.prompt_loop(gm, speak_gm=True)

    init.assert_called_once()
    speak.assert_called_once_with("engine", "First response")
    assert gm.inputs == ["Hello"]


def test_initialize_voice_output_selects_korean_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The TTS helper should pick a Korean voice when one is available."""

    engine = mock.MagicMock()
    engine.getProperty.side_effect = [
        [
            SimpleNamespace(id="english", languages=[b"en_US"]),
            SimpleNamespace(id="korean", languages=[b"ko_KR"]),
        ],
        180,
    ]

    fake_pyttsx3 = SimpleNamespace(init=mock.Mock(return_value=engine))
    monkeypatch.setattr(main, "_pyttsx3", fake_pyttsx3)

    result = main._initialize_voice_output()

    assert result is engine
    fake_pyttsx3.init.assert_called_once_with()
    assert mock.call("volume", 1.0) in engine.setProperty.call_args_list
    assert any(call.args[0] == "rate" for call in engine.setProperty.call_args_list)
    assert mock.call("voice", "korean") in engine.setProperty.call_args_list


def test_initialize_voice_output_handles_missing_korean_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no Korean voice exists, the engine should still be configured safely."""

    engine = mock.MagicMock()
    engine.getProperty.side_effect = [
        [SimpleNamespace(id="english", languages=["en_US"])],
        200,
    ]

    fake_pyttsx3 = SimpleNamespace(init=mock.Mock(return_value=engine))
    monkeypatch.setattr(main, "_pyttsx3", fake_pyttsx3)

    result = main._initialize_voice_output()

    assert result is engine
    fake_pyttsx3.init.assert_called_once_with()
    assert mock.call("volume", 1.0) in engine.setProperty.call_args_list
    assert any(call.args[0] == "rate" for call in engine.setProperty.call_args_list)


def test_initialize_voice_output_uses_korean_named_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voices whose names contain Korean text should be selected."""

    engine = mock.MagicMock()
    engine.getProperty.side_effect = [
        [
            SimpleNamespace(id="voice-1", name="English", languages=[]),
            SimpleNamespace(id="voice-2", name="한국어 음성", languages=[]),
        ],
        190,
    ]

    fake_pyttsx3 = SimpleNamespace(init=mock.Mock(return_value=engine))
    monkeypatch.setattr(main, "_pyttsx3", fake_pyttsx3)

    result = main._initialize_voice_output()

    assert result is engine
    fake_pyttsx3.init.assert_called_once_with()
    assert mock.call("volume", 1.0) in engine.setProperty.call_args_list
    assert any(call.args[0] == "rate" for call in engine.setProperty.call_args_list)
    assert mock.call("voice", "voice-2") in engine.setProperty.call_args_list


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
