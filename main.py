"""Command line interface for playing a LangChain powered TRPG."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from langchain.chat_models import ChatOpenAI

from trpg import GameMaster, create_default_game_master


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide an API key to talk to the LLM."
        )

    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)


def build_game_master(model: str, temperature: float) -> GameMaster:
    llm = build_llm(model=model, temperature=temperature)
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
        default="gpt-3.5-turbo",
        help="OpenAI chat model to use via LangChain (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature to use for the LLM (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        gm = build_game_master(model=args.model, temperature=args.temperature)
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
