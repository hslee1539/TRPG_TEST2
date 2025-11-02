"""Web server exposing a browser-based TRPG experience."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from flask import Flask, jsonify, render_template, request

from trpg import GameMaster, create_default_game_master


class SimpleLocalLLM:
    """A lightweight storyteller that keeps the game offline."""

    def __init__(self) -> None:
        self.turn = 0

    def invoke(self, messages):  # type: ignore[override]
        """Return a friendly narrative response based on player input."""

        self.turn += 1
        prompt = str(messages[-1].content)
        player_input = self._extract_player_input(prompt)
        return self._build_response(player_input)

    @staticmethod
    def _extract_player_input(prompt: str) -> str:
        match = re.search(r"The player says:\s*(.*?)\nContinue", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        return prompt.strip()

    def _build_response(self, player_input: str) -> str:
        if not player_input:
            player_input = "아무 말도 하지 않는 것 같군요."

        openings = [
            "바람이 살짝 흔들리며 주변을 감싼다.",
            "머나먼 종소리가 어렴풋이 울려 퍼진다.",
            "낡은 지도 위에 빛무리가 스쳐 지나간다.",
            "따뜻한 햇살이 당신의 어깨를 감싸 안는다.",
        ]
        opening = openings[self.turn % len(openings)]
        return (
            f"{opening} 당신의 선택 '{player_input}' 에 따라 이야기가 새롭게 전개된다. "
            "새로운 단서와 친구를 찾기 위해 주변을 살펴보는 것이 어떨까요?"
        )


@dataclass
class Session:
    """Container for a single browser game session."""

    identifier: str
    game_master: GameMaster
    history: List[Dict[str, str]] = field(default_factory=list)

    def record(self, role: str, message: str) -> None:
        self.history.append({"role": role, "message": message})


app = Flask(__name__)
_sessions: Dict[str, Session] = {}


def _create_game_master() -> GameMaster:
    llm = SimpleLocalLLM()
    return create_default_game_master(llm)


@app.post("/api/session")
def create_session():
    identifier = uuid.uuid4().hex
    gm = _create_game_master()
    session = Session(identifier=identifier, game_master=gm)
    greeting = (
        "안녕하세요! 이곳은 소규모 판타지 마을입니다. "
        "당신의 목표는 마음이 가는 대로 모험을 펼치며 작은 영웅이 되는 것입니다."
    )
    session.record("gm", greeting)
    gm.state.add_fact(f"GM: {greeting}")
    _sessions[identifier] = session
    return jsonify(
        {
            "sessionId": identifier,
            "history": session.history,
            "scene": gm.render_scene(width=50),
        }
    )


@app.post("/api/session/<session_id>/message")
def send_message(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "세션을 찾을 수 없습니다."}), 404

    data = request.get_json(silent=True) or {}
    player_input = str(data.get("message", "")).strip()
    if not player_input:
        return jsonify({"error": "메시지를 입력해 주세요."}), 400

    session.record("player", player_input)
    gm_response = session.game_master.respond(player_input)
    session.record("gm", gm_response)
    return jsonify(
        {
            "history": session.history,
            "gm": gm_response,
            "scene": session.game_master.render_scene(width=50),
        }
    )


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
