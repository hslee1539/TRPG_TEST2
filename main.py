"""Command line interface for playing a LangChain powered TRPG."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from langchain.chat_models import ChatOpenAI

from trpg import GameMaster, create_default_game_master
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


def prompt_loop(gm: GameMaster, initial_prompt: Optional[str] = None) -> None:
    if initial_prompt:
        print(initial_prompt)
    print("Type 'quit' to end the session.\n")

    while True:
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
    prompt_loop(gm, initial_prompt=intro)
    return 0


if __name__ == "__main__":
    sys.exit(main())
