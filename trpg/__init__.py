"""TRPG package allowing an LLM to act as the game master."""

from .game_master import GameMaster, GameState, create_default_game_master

__all__ = [
    "GameMaster",
    "GameState",
    "create_default_game_master",
]
