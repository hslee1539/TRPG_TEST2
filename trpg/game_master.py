"""Core logic for running a TRPG session powered by an LLM via LangChain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


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


class GameMaster:
    """Wraps an LLMChain to orchestrate TRPG style conversations."""

    def __init__(
        self,
        chain: LLMChain,
        state: Optional[GameState] = None,
        initial_facts: Optional[Iterable[str]] = None,
    ) -> None:
        self.chain = chain
        self.state = state or GameState()
        if initial_facts:
            for fact in initial_facts:
                self.state.add_fact(fact)

    def build_prompt(self, player_input: str) -> dict:
        """Build the structured input expected by the underlying LangChain."""

        return {
            "player_input": player_input,
            "facts": self.state.to_bullet_list(),
        }

    def respond(self, player_input: str) -> str:
        """Send the player's input to the LLM and update the shared state."""

        prompt_variables = self.build_prompt(player_input)
        response = self.chain.predict(**prompt_variables)
        self.state.add_fact(f"Player: {player_input}")
        self.state.add_fact(f"GM: {response}")
        return response


def create_default_game_master(llm) -> GameMaster:
    """Create a GameMaster with a story focused system prompt."""

    system_template = (
        "You are a warm, collaborative tabletop role-playing game master. "
        "Guide the player through an exciting narrative, respond to their "
        "actions, describe the world vividly, and ask follow-up questions "
        "to keep the story moving. Keep responses under 200 words."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Facts so far:\n{facts}\n\n"
                "The player says: {player_input}\n"
                "Continue the story, incorporating relevant facts.",
            ),
        ]
    )

    memory = ConversationBufferMemory(return_messages=True)

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return GameMaster(chain)
