"""Web server exposing a browser-based TRPG experience."""
from __future__ import annotations

import argparse
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from trpg import GameMaster, create_default_game_master


LLMFactory = Callable[[], Any]


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

    def draw_scene(self, *, gm_text: str, player_input: str, facts: List[str]) -> str:
        theme = _select_scene_theme(gm_text, player_input, "\n".join(facts))
        palette = {
            "town": ("#f97316", "#facc15"),
            "forest": ("#16a34a", "#4ade80"),
            "dungeon": ("#4b5563", "#9ca3af"),
            "castle": ("#6366f1", "#a855f7"),
        }
        primary, accent = palette.get(theme, palette["town"])

        return """
<svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" role="img">""".strip() + (
            ""
            "\n  <defs>\n"
            "    <linearGradient id=\"scene-gradient\" x1=\"0%\" y1=\"0%\" x2=\"0%\" y2=\"100%\">\n"
            f"      <stop offset=\"0%\" stop-color=\"{primary}\" />\n"
            f"      <stop offset=\"100%\" stop-color=\"{accent}\" />\n"
            "    </linearGradient>\n"
            "  </defs>\n"
            "  <rect x=\"0\" y=\"0\" width=\"400\" height=\"300\" fill=\"url(#scene-gradient)\" rx=\"24\" />\n"
            "  <g fill=\"rgba(255,255,255,0.18)\">\n"
            "    <circle cx=\"60\" cy=\"70\" r=\"28\" />\n"
            "    <circle cx=\"340\" cy=\"80\" r=\"32\" />\n"
            "    <circle cx=\"210\" cy=\"220\" r=\"36\" />\n"
            "  </g>\n"
            f"{_render_scene_layer(theme, primary, accent)}\n"
            "</svg>"
        )


@dataclass
class Session:
    """Container for a single browser game session."""

    identifier: str
    game_master: GameMaster
    history: List[Dict[str, str]] = field(default_factory=list)
    scene_image: str = ""
    scene_alt_text: str = "현재 장면을 표현한 일러스트"

    def record(self, role: str, message: str) -> None:
        self.history.append({"role": role, "message": message})

app = Flask(__name__)
_sessions: Dict[str, Session] = {}


def _default_llm_factory() -> SimpleLocalLLM:
    return SimpleLocalLLM()


_llm_factory: LLMFactory = _default_llm_factory


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_svg(name: str) -> str:
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


SCENE_IMAGE_PLACEHOLDER = _load_svg("scene-placeholder.svg")

_SCENE_KEYWORDS = [
    (
        "forest",
        {
            "숲",
            "수풀",
            "나무",
            "정글",
            "forest",
            "woods",
            "grove",
        },
    ),
    (
        "dungeon",
        {
            "던전",
            "지하",
            "동굴",
            "폐허",
            "dungeon",
            "cave",
            "crypt",
        },
    ),
    (
        "castle",
        {
            "성",
            "성벽",
            "궁전",
            "castle",
            "fortress",
            "palace",
        },
    ),
    (
        "town",
        {
            "마을",
            "도시",
            "시장",
            "거리",
            "town",
            "village",
            "city",
        },
    ),
]


def _normalise_text_for_scene(text: str) -> str:
    return text.lower()


def _select_scene_theme(*texts: str, default: str = "town") -> str:
    for text in texts:
        if not text:
            continue
        normalised = _normalise_text_for_scene(text)
        for theme, keywords in _SCENE_KEYWORDS:
            if any(keyword.lower() in normalised for keyword in keywords):
                return theme
    return default


def _render_scene_layer(theme: str, primary: str, accent: str) -> str:
    """Return layered SVG markup for the requested theme."""

    if theme == "forest":
        return f"""
  <g>
    <rect x="0" y="210" width="400" height="90" fill="#14532d" opacity="0.85" />
    <g>
      <rect x="60" y="170" width="18" height="80" fill="#4d2c1d" />
      <polygon points="69,110 35,190 103,190" fill="{accent}" opacity="0.9" />
    </g>
    <g>
      <rect x="170" y="190" width="16" height="70" fill="#4d2c1d" />
      <polygon points="178,140 146,210 210,210" fill="{primary}" opacity="0.85" />
    </g>
    <g>
      <rect x="270" y="180" width="14" height="80" fill="#4d2c1d" />
      <polygon points="277,130 248,200 308,200" fill="{accent}" opacity="0.85" />
    </g>
    <circle cx="320" cy="90" r="36" fill="#fde68a" opacity="0.45" />
  </g>
""".strip()

    if theme == "dungeon":
        return f"""
  <g>
    <rect x="0" y="200" width="400" height="100" fill="#1f2937" opacity="0.9" />
    <rect x="130" y="120" width="140" height="160" rx="24" fill="#374151" />
    <path d="M150 220 Q200 160 250 220" fill="#111827" />
    <path d="M150 220 Q200 260 250 220" fill="#0f172a" opacity="0.7" />
    <g fill="{accent}">
      <circle cx="140" cy="160" r="10" />
      <rect x="138" y="165" width="4" height="26" />
      <circle cx="260" cy="160" r="10" />
      <rect x="258" y="165" width="4" height="26" />
    </g>
    <rect x="180" y="240" width="40" height="40" fill="{primary}" opacity="0.6" />
  </g>
""".strip()

    if theme == "castle":
        return f"""
  <g>
    <rect x="0" y="210" width="400" height="90" fill="#312e81" opacity="0.95" />
    <rect x="60" y="130" width="60" height="170" fill="#4c1d95" />
    <rect x="280" y="130" width="60" height="170" fill="#4c1d95" />
    <rect x="120" y="180" width="160" height="120" fill="{primary}" />
    <rect x="150" y="210" width="40" height="90" fill="#312e81" />
    <rect x="210" y="210" width="40" height="90" fill="#312e81" />
    <polygon points="90,130 90,100 120,100 120,130" fill="{accent}" />
    <polygon points="310,130 310,100 340,100 340,130" fill="{accent}" />
    <polygon points="200,120 180,90 220,90" fill="#c4b5fd" />
    <path d="M110 180 L140 150 L170 180" fill="{primary}" opacity="0.8" />
    <path d="M230 180 L260 150 L290 180" fill="{primary}" opacity="0.8" />
    <circle cx="200" cy="120" r="16" fill="#fef3c7" opacity="0.6" />
  </g>
""".strip()

    # default town scene
    return f"""
  <g>
    <rect x="0" y="220" width="400" height="80" fill="#92400e" opacity="0.9" />
    <g>
      <rect x="70" y="170" width="80" height="110" fill="{primary}" />
      <polygon points="70,170 110,120 150,170" fill="{accent}" />
      <rect x="95" y="220" width="30" height="60" fill="#78350f" />
      <rect x="80" y="190" width="24" height="24" fill="#fed7aa" />
      <rect x="116" y="190" width="24" height="24" fill="#fed7aa" />
    </g>
    <g>
      <rect x="220" y="190" width="90" height="90" fill="#fb7185" />
      <polygon points="220,190 265,140 310,190" fill="{accent}" />
      <rect x="245" y="230" width="30" height="50" fill="#9f1239" />
      <rect x="232" y="205" width="20" height="20" fill="#fecdd3" />
      <rect x="278" y="205" width="20" height="20" fill="#fecdd3" />
    </g>
    <path d="M0 250 Q200 230 400 270" fill="#fde68a" opacity="0.35" />
    <circle cx="320" cy="90" r="26" fill="#fde68a" opacity="0.5" />
  </g>
""".strip()

def _is_valid_svg(svg: str | None) -> bool:
    if not svg:
        return False
    cleaned = svg.strip().lower()
    return cleaned.startswith("<svg") and "</svg>" in cleaned


def configure_llm(factory: LLMFactory) -> None:
    """Configure the callable used to create LLM instances for new sessions."""

    global _llm_factory
    _llm_factory = factory


def use_llm_instance(llm: Any) -> None:
    """Pin the server to an already-instantiated LLM object."""

    configure_llm(lambda: llm)


DEFAULT_LM_STUDIO_API_BASE = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"


class LMStudioStoryteller:
    """Wrapper around the LM Studio OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without openai installed
            raise RuntimeError(
                "openai 패키지를 찾을 수 없습니다. 'pip install openai' 명령으로 설치한 뒤 다시 시도해 주세요."
            ) from exc

        resolved_api_base = api_base or os.getenv(
            "LM_STUDIO_API_BASE", DEFAULT_LM_STUDIO_API_BASE
        )
        resolved_api_key = api_key or os.getenv(
            "LM_STUDIO_API_KEY", DEFAULT_LM_STUDIO_API_KEY
        )

        self._client = OpenAI(base_url=resolved_api_base, api_key=resolved_api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _normalise_role(message: Any) -> str:
        role = getattr(message, "type", None) or getattr(message, "role", None)
        role = str(role or "user").lower()
        if role in {"ai", "assistant"}:
            return "assistant"
        if role in {"human", "user"}:
            return "user"
        if role == "system":
            return "system"
        return "user"

    def _build_payload(self, messages: List[Any]) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        for message in messages:
            payload.append(
                {
                    "role": self._normalise_role(message),
                    "content": GameMaster._message_content(message),
                }
            )
        return payload

    def invoke(self, messages: List[Any]):  # type: ignore[override]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=self._build_payload(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content if response.choices else ""
        return str(content or "").strip()

    def draw_scene(self, *, gm_text: str, player_input: str, facts: List[str]) -> str:
        fact_summary = "\n".join(f"- {fact}" for fact in facts[-6:]) or "(기록 없음)"
        system_prompt = (
            "당신은 SVG 일러스트레이터입니다."
            " 응답은 반드시 유효한 단일 <svg>...</svg> 마크업이어야 합니다."
            " 배경, 기본 색상과 간단한 도형을 활용해 장면을 묘사하세요."
            " 스크립트, 외부 참조, 설명 텍스트는 포함하지 마세요."
            " 400x300 뷰박스를 사용하고 텍스트는 3줄 이내로 유지하세요."
        )
        user_prompt = (
            "최근 GM 묘사: {gm}\n"
            "플레이어 행동: {player}\n"
            "중요 사실:\n{facts}\n\n"
            "위 정보를 반영한 간결한 SVG를 그려주세요."
        ).format(
            gm=gm_text or "(없음)",
            player=player_input or "(없음)",
            facts=fact_summary,
        )
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=min(self.temperature, 0.6),
            max_tokens=min(self.max_tokens * 2, 1200),
        )
        svg = response.choices[0].message.content if response.choices else ""
        svg_text = str(svg or "").strip()
        if not _is_valid_svg(svg_text):
            raise ValueError("LM Studio가 SVG 응답을 반환하지 않았습니다.")
        return svg_text


def _create_game_master() -> GameMaster:
    llm = _llm_factory()
    return create_default_game_master(llm)


def _generate_scene_svg(session: Session, *, gm_text: str, player_input: str) -> str:
    llm = session.game_master.llm
    facts = session.game_master.state.facts
    if not hasattr(llm, "draw_scene"):
        raise RuntimeError("현재 LLM은 draw_scene 기능을 지원하지 않습니다.")

    svg = llm.draw_scene(
        gm_text=gm_text,
        player_input=player_input,
        facts=facts,
    )
    if not _is_valid_svg(svg):
        raise ValueError("LLM이 유효한 SVG를 반환하지 않았습니다.")
    return svg


def configure_lmstudio_model(
    model_name: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    """Configure the server to use LM Studio via its OpenAI-compatible API."""

    cache: Dict[str, Any] = {}

    def factory() -> Any:
        if "llm" not in cache:
            cache["llm"] = LMStudioStoryteller(
                model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=api_base,
                api_key=api_key,
            )
        return cache["llm"]

    configure_llm(factory)


@app.post("/api/session")
def create_session():
    identifier = uuid.uuid4().hex
    gm = _create_game_master()
    session = Session(
        identifier=identifier,
        game_master=gm,
        scene_image=SCENE_IMAGE_PLACEHOLDER,
    )
    greeting = (
        "안녕하세요! 이곳은 소규모 판타지 마을입니다. "
        "당신의 목표는 마음이 가는 대로 모험을 펼치며 작은 영웅이 되는 것입니다."
    )
    session.record("gm", greeting)
    gm.state.add_fact(f"GM: {greeting}")
    try:
        session.scene_image = _generate_scene_svg(session, gm_text=greeting, player_input="")
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "장면 이미지를 생성하지 못했습니다.",
                    "details": str(exc),
                }
            ),
            500,
        )
    session.scene_alt_text = greeting
    _sessions[identifier] = session
    return jsonify(
        {
            "sessionId": identifier,
            "history": session.history,
            "scene": gm.render_scene(width=50),
            "sceneImage": session.scene_image,
            "sceneImageAlt": session.scene_alt_text,
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
    try:
        session.scene_image = _generate_scene_svg(
            session,
            gm_text=gm_response,
            player_input=player_input,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "장면 이미지를 생성하지 못했습니다.",
                    "details": str(exc),
                }
            ),
            500,
        )
    session.scene_alt_text = gm_response or session.scene_alt_text
    return jsonify(
        {
            "history": session.history,
            "gm": gm_response,
            "scene": session.game_master.render_scene(width=50),
            "sceneImage": session.scene_image,
            "sceneImageAlt": session.scene_alt_text,
        }
    )


@app.get("/")
def index():
    return render_template("index.html", placeholder_svg=SCENE_IMAGE_PLACEHOLDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TRPG web server.")
    parser.add_argument("--model", help="LM Studio에서 실행 중인 모델 이름")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--api-base",
        help="LM Studio OpenAI 호환 서버의 주소 (기본: 환경변수 또는 http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--api-key",
        help="LM Studio OpenAI 호환 서버에 전달할 API 키 (기본: 환경변수 또는 lm-studio)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    if args.model:
        configure_lmstudio_model(
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_base=args.api_base,
            api_key=args.api_key,
        )

    app.run(host=args.host, port=args.port, debug=False)
