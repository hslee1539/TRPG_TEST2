"""Command line interface for playing a LangChain powered TRPG."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Tuple

try:  # pragma: no cover - import paths differ across langchain versions
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - fallback for alternate package layouts
    try:
        from langchain_community.chat_models import ChatOpenAI  # type: ignore
    except ImportError:  # pragma: no cover - fallback for legacy langchain
        from langchain.chat_models import ChatOpenAI  # type: ignore

from trpg import GameMaster, create_default_game_master

try:  # pragma: no cover - optional dependency for voice input
    import speech_recognition as _speech_recognition
except ImportError:  # pragma: no cover - voice input is optional
    _speech_recognition = None
DEFAULT_LM_STUDIO_API_BASE = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"


def build_llm(
    model: str,
    temperature: float,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    resolved_api_base = api_base or os.getenv(
        "LM_STUDIO_API_BASE", DEFAULT_LM_STUDIO_API_BASE
    )
    resolved_api_key = api_key or os.getenv(
        "LM_STUDIO_API_KEY", DEFAULT_LM_STUDIO_API_KEY
    )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=resolved_api_key,
        openai_api_base=resolved_api_base,
    )


def build_game_master(
    model: str,
    temperature: float,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> GameMaster:
    llm = build_llm(
        model=model,
        temperature=temperature,
        api_base=api_base,
        api_key=api_key,
    )
    return create_default_game_master(llm)


def _initialize_voice_capture() -> Tuple[Any, Any]:
    """Prepare objects required to capture audio from the microphone."""

    if _speech_recognition is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "SpeechRecognition is not installed. Install it to use voice input."
        )

    recognizer = _speech_recognition.Recognizer()
    try:
        microphone = _speech_recognition.Microphone()
    except OSError as exc:
        raise RuntimeError(f"Unable to access a microphone: {exc}") from exc

    # Calibrate for ambient noise to improve recognition accuracy.
    with microphone as source:  # pragma: no cover - requires audio hardware
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

    return recognizer, microphone


def _capture_voice_input(recognizer: Any, microphone: Any) -> Optional[str]:
    """Listen to the microphone and return transcribed speech."""

    if _speech_recognition is None:  # pragma: no cover - runtime guard
        raise RuntimeError("SpeechRecognition is not installed.")

    print("🎤 Speak now (press Ctrl+C to cancel)...")
    with microphone as source:  # pragma: no cover - requires audio hardware
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
    except _speech_recognition.UnknownValueError:
        print("Sorry, I didn't catch that. Let's try again.\n")
        return None
    except _speech_recognition.RequestError as exc:
        raise RuntimeError(f"Speech recognition service error: {exc}") from exc

    return text


def prompt_loop(
    gm: GameMaster,
    initial_prompt: Optional[str] = None,
    *,
    input_mode: str = "text",
) -> None:
    if initial_prompt:
        print(initial_prompt)
    if input_mode == "voice":
        print("Say 'quit' to end the session.\n")
    else:
        print("Type 'quit' to end the session.\n")

    voice_resources: Optional[Tuple[Any, Any]] = None
    if input_mode == "voice":
        try:
            voice_resources = _initialize_voice_capture()
        except RuntimeError as exc:
            print(f"Voice input is unavailable: {exc}", file=sys.stderr)
            return

    while True:
        if input_mode == "voice":
            assert voice_resources is not None  # Satisfy type checkers.
            recognizer, microphone = voice_resources
            try:
                user_input = _capture_voice_input(recognizer, microphone)
            except KeyboardInterrupt:
                print()
                break
            except RuntimeError as exc:
                print(f"Voice input error: {exc}", file=sys.stderr)
                break

            if user_input is None:
                continue

            print(f"You: {user_input}")
        else:
            try:
                user_input = input("You: ")
            except EOFError:
                print()  # Provide a trailing newline when exiting with Ctrl+D.
                break

        if user_input.strip().lower() in {"quit", "exit"}:
            break

        try:
            response = gm.respond(user_input)
        except Exception as exc:  # pragma: no cover - runtime errors bubble to CLI
            print(f"Error communicating with the LLM: {exc}", file=sys.stderr)
            break

        print(f"GM: {response}\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        help="LM Studio chat model to use via LangChain (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature to use for the LLM (default: %(default)s)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "Base URL for the LM Studio OpenAI-compatible server. "
            "Defaults to the value of LM_STUDIO_API_BASE or "
            f"{DEFAULT_LM_STUDIO_API_BASE}."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for the LM Studio OpenAI-compatible server. "
            "Defaults to the value of LM_STUDIO_API_KEY or "
            f"{DEFAULT_LM_STUDIO_API_KEY}."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=("text", "voice"),
        default="text",
        help="How to capture player input (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        gm = build_game_master(
            model=args.model,
            temperature=args.temperature,
            api_base=args.api_base,
            api_key=args.api_key,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    intro = (
        "Welcome to the LangChain powered TRPG! The LLM will lead the story "
        "as your game master."
    )
    prompt_loop(gm, initial_prompt=intro, input_mode=args.input_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
