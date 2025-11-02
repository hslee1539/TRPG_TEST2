"""Core logic for running a TRPG session powered by an LLM via LangChain."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import re
import textwrap
from typing import Any, Iterable, List, Optional, Sequence


def _resolve_attr(name: str, modules: Sequence[str]) -> Any:
    """Return the requested attribute from the first importable module."""

    last_exc: Optional[Exception] = None
    for module_name in modules:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive fallback
            last_exc = exc
            continue
        try:
            return getattr(module, name)
        except AttributeError as exc:  # pragma: no cover - defensive fallback
            last_exc = exc
            continue
    modules_display = ", ".join(modules)
    raise ImportError(f"Could not import {name} from any of: {modules_display}") from last_exc


BaseMessage = _resolve_attr(
    "BaseMessage", ["langchain_core.messages", "langchain.schema"]
)
SystemMessage = _resolve_attr(
    "SystemMessage", ["langchain_core.messages", "langchain.schema"]
)
HumanMessage = _resolve_attr(
    "HumanMessage", ["langchain_core.messages", "langchain.schema"]
)
AIMessage = _resolve_attr("AIMessage", ["langchain_core.messages", "langchain.schema"])


@dataclass
class GameState:
    """Lightweight container holding story beats and world facts."""

    facts: List[str] = field(default_factory=list)

    def add_fact(self, fact: str) -> None:
        """Store a new fact about the world or the ongoing scene."""

        fact = fact.strip()
        if fact:
            self.facts.append(fact)

    def to_bullet_list(self) -> str:
        """Represent the stored facts as a human readable bullet list."""

        if not self.facts:
            return "(no established facts yet)"
        return "\n".join(f"- {fact}" for fact in self.facts)

    def render_scene(self, *, width: int = 60) -> str:
        """Render an ASCII "image" summarising the known facts."""

        width = max(20, width)
        border = "+" + "-" * (width + 2) + "+"

        if not self.facts:
            message = "(no established facts yet)"
            centered = message.center(width)
            body_lines = [f"| {centered} |"]
        else:
            body_lines: List[str] = []
            bullet_space = 2  # account for bullet and a following space
            wrap_width = width - bullet_space
            for fact in self.facts:
                wrapped = textwrap.wrap(fact, wrap_width) or [""]
                for index, segment in enumerate(wrapped):
                    bullet = "• " if index == 0 else "  "
                    padded = (bullet + segment).ljust(width)
                    body_lines.append(f"| {padded} |")

        return "\n".join([border, *body_lines, border])


class GameMaster:
    """Coordinate a LangChain chat model to run a TRPG session."""

    def __init__(
        self,
        llm: Any,
        *,
        state: Optional[GameState] = None,
        initial_facts: Optional[Iterable[str]] = None,
        system_template: str,
    ) -> None:
        self.llm = llm
        self.state = state or GameState()
        self._system_message = SystemMessage(content=system_template)
        self._chat_history: List[BaseMessage] = []
        if initial_facts:
            for fact in initial_facts:
                self.state.add_fact(fact)

    def build_messages(self, player_input: str) -> List[BaseMessage]:
        """Create the LangChain messages used for the next LLM call."""

        facts = self.state.to_bullet_list()
        prompt = (
            "Facts so far:\n"
            f"{facts}\n\n"
            "The player says: {player_input}\n"
            "Continue the story, incorporating relevant facts."
        )
        player_message = HumanMessage(content=prompt.format(player_input=player_input))
        return [self._system_message, *self._chat_history, player_message]

    def _call_llm(self, messages: List[BaseMessage]) -> Any:
        """Invoke the backing chat model with defensive fallbacks."""

        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(messages)
        if hasattr(self.llm, "predict_messages"):
            return self.llm.predict_messages(messages)  # pragma: no cover - legacy path
        if hasattr(self.llm, "__call__"):
            return self.llm(messages)  # pragma: no cover - highly defensive
        raise AttributeError("LLM does not support invoke, predict_messages, or __call__")

    @staticmethod
    def _message_content(message: Any) -> str:
        """Extract text content from LangChain messages or strings."""

        if hasattr(message, "content"):
            return str(message.content)
        return str(message)

    def respond(self, player_input: str) -> str:
        """Send the player's input to the LLM and update the shared state."""

        messages = self.build_messages(player_input)
        raw_response = self._call_llm(messages)
        response_text = self._strip_hidden_thoughts(
            self._message_content(raw_response)
        )

        self._chat_history.extend(
            [HumanMessage(content=player_input), AIMessage(content=response_text)]
        )
        self.state.add_fact(f"Player: {player_input}")
        self.state.add_fact(f"GM: {response_text}")
        return response_text

    @staticmethod
    def _strip_hidden_thoughts(text: str) -> str:
        """Remove hidden reasoning tags (e.g. ``<think>``) from model output."""

        pattern = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
        cleaned = pattern.sub("", text)
        return cleaned.strip()

    def render_scene(self, *, width: int = 60) -> str:
        """Expose the ASCII representation of the tracked facts."""

        return self.state.render_scene(width=width)


def create_default_game_master(llm) -> GameMaster:
    """Create a GameMaster with a story focused system prompt."""

    system_template = (
        "You are a warm, collaborative tabletop role-playing game master. "
        "Guide the player through an exciting narrative, respond to their "
        "actions, describe the world vividly, and ask follow-up questions "
        "to keep the story moving. Keep responses under 200 words and "
        "answer in Korean."
    )
    return GameMaster(llm, system_template=system_template)
