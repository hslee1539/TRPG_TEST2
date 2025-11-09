"""Web server exposing a browser-based TRPG experience."""
from __future__ import annotations

import argparse
import base64
import io
import inspect
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, runtime_checkable

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
        svg = _build_keyword_scene_svg(gm_text=gm_text, player_input=player_input, facts=facts)
        return svg


@runtime_checkable
class SceneGenerator(Protocol):
    """Protocol describing an object capable of generating scene imagery."""

    def generate(self, *, gm_text: str, player_input: str, facts: List[str]) -> str:
        """Return a data URI containing the rendered scene image."""


SceneGeneratorFactory = Callable[[], SceneGenerator]


@dataclass
class Session:
    """Container for a single browser game session."""

    identifier: str
    game_master: GameMaster
    scene_generator: SceneGenerator
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


def _encode_svg_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _encode_image_to_data_uri(image: Any) -> str:
    """Encode a PIL image or array-like object into a PNG data URI."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - only triggered without Pillow
        raise RuntimeError(
            "Pillow 패키지가 필요합니다. 'pip install Pillow' 명령으로 설치해 주세요."
        ) from exc

    if hasattr(image, "to_pil"):
        image = image.to_pil()
    elif not isinstance(image, Image.Image):
        # Accept array-like results and convert to PIL.Image
        if hasattr(image, "__array__"):
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover - requires numpy at runtime
                raise RuntimeError(
                    "numpy 패키지가 필요합니다. 'pip install numpy' 명령으로 설치해 주세요."
                ) from exc

            array = image.__array__()
            image = Image.fromarray(np.asarray(array).clip(0, 255).astype("uint8"))
        else:  # pragma: no cover - defensive fallback when pipeline returns bytes
            raise TypeError("Stable Diffusion 결과를 이미지로 변환할 수 없습니다.")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


SCENE_PLACEHOLDER_SVG = _load_svg("scene-placeholder.svg")
SCENE_IMAGE_PLACEHOLDER = _encode_svg_data_uri(SCENE_PLACEHOLDER_SVG)


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
      <rect x="280" y="180" width="14" height="75" fill="#4d2c1d" />
      <polygon points="287,130 250,195 320,195" fill="{accent}" opacity="0.88" />
    </g>
  </g>
  <g fill="rgba(255, 255, 255, 0.12)">
    <circle cx="120" cy="120" r="42" />
    <circle cx="300" cy="70" r="30" />
  </g>
""".strip()

    if theme == "dungeon":
        return f"""
  <g>
    <rect x="0" y="200" width="400" height="100" fill="#1f2937" opacity="0.9" />
    <rect x="80" y="160" width="240" height="90" rx="12" fill="{primary}" opacity="0.85" />
    <g fill="rgba(0, 0, 0, 0.2)">
      <rect x="100" y="180" width="30" height="40" />
      <rect x="150" y="180" width="30" height="40" />
      <rect x="200" y="180" width="30" height="40" />
      <rect x="250" y="180" width="30" height="40" />
    </g>
    <circle cx="60" cy="90" r="28" fill="{accent}" opacity="0.6" />
    <circle cx="340" cy="70" r="24" fill="{accent}" opacity="0.45" />
  </g>
""".strip()

    if theme == "castle":
        return f"""
  <g>
    <rect x="20" y="160" width="360" height="120" fill="{primary}" opacity="0.85" />
    <g>
      <rect x="40" y="120" width="60" height="80" fill="#312e81" />
      <rect x="300" y="120" width="60" height="80" fill="#312e81" />
      <rect x="160" y="100" width="80" height="110" fill="#3730a3" />
      <polygon points="200,60 150,100 250,100" fill="{accent}" opacity="0.9" />
    </g>
    <rect x="185" y="150" width="30" height="60" rx="8" fill="#f5f3ff" opacity="0.85" />
    <circle cx="320" cy="90" r="26" fill="{accent}" opacity="0.65" />
  </g>
  <path d="M0 240 Q200 210 400 260" fill="#ede9fe" opacity="0.35" />
""".strip()

    return f"""
  <g>
    <rect x="0" y="220" width="400" height="80" fill="#f97316" opacity="0.35" />
    <g>
      <rect x="60" y="160" width="70" height="70" rx="12" fill="{primary}" opacity="0.92" />
      <rect x="74" y="185" width="24" height="35" fill="#fde68a" opacity="0.9" />
    </g>
    <g>
      <rect x="240" y="150" width="90" height="80" rx="14" fill="{accent}" opacity="0.88" />
      <rect x="260" y="185" width="24" height="35" fill="#fef3c7" opacity="0.92" />
    </g>
    <path d="M50 250 Q200 230 350 260" stroke="#fcd34d" stroke-width="6" fill="none" opacity="0.75" />
    <circle cx="90" cy="110" r="26" fill="#fde68a" opacity="0.5" />
  </g>
""".strip()


def _build_keyword_scene_svg(*, gm_text: str, player_input: str, facts: Iterable[str]) -> str:
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


class KeywordSceneGenerator:
    """Default scene generator that returns themed SVG data URIs."""

    def generate(self, *, gm_text: str, player_input: str, facts: List[str]) -> str:
        svg = _build_keyword_scene_svg(gm_text=gm_text, player_input=player_input, facts=facts)
        return _encode_svg_data_uri(svg)


def _default_scene_generator_factory() -> SceneGenerator:
    return KeywordSceneGenerator()


_scene_generator_factory: SceneGeneratorFactory = _default_scene_generator_factory


DEFAULT_SD_MODEL_ID = "mlx-community/stable-diffusion-v1-5-diffusers"


def _is_valid_data_uri(data: str | None) -> bool:
    if not data:
        return False
    return data.startswith("data:image/")


def configure_llm(factory: LLMFactory) -> None:
    """Configure the callable used to create LLM instances for new sessions."""

    global _llm_factory
    _llm_factory = factory


def use_llm_instance(llm: Any) -> None:
    """Pin the server to an already-instantiated LLM object."""

    configure_llm(lambda: llm)


def configure_scene_generator(factory: SceneGeneratorFactory) -> None:
    """Configure the factory used to create scene generators for sessions."""

    global _scene_generator_factory
    _scene_generator_factory = factory


def use_scene_generator(generator: SceneGenerator) -> None:
    """Pin the server to an already-instantiated scene generator."""

    configure_scene_generator(lambda: generator)


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


def _call_with_supported_kwargs(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call the function with only the keyword arguments it supports."""

    signature = inspect.signature(func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return func(*args, **filtered_kwargs)


class MLXStableDiffusionSceneGenerator:
    """Generate scene imagery using MLX Stable Diffusion with quantised weights."""

    def __init__(
        self,
        model_path: str,
        *,
        quantize: bool = True,
        steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
    ) -> None:
        try:
            from mlx_examples.stable_diffusion import StableDiffusionPipeline
        except ImportError as exc:  # pragma: no cover - exercised only without mlx installed
            raise RuntimeError(
                "mlx-examples 모듈을 찾을 수 없습니다. 먼저 'pip install mlx Pillow numpy'를 실행한 뒤, "
                "https://github.com/ml-explore/mlx-examples 저장소를 클론하고 Stable Diffusion 예제 폴더의 요구 사항을 "
                "'pip install -r stable_diffusion/requirements.txt'로 설치했는지, 그리고 \"export PYTHONPATH='$(pwd):${PYTHONPATH}'\" 명령으로 "
                "경로를 추가했는지 확인하세요."
            ) from exc

        self.steps = steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = (
            negative_prompt
            or "blurry, distorted, low quality, text, watermark, signature"
        )
        self.width = width
        self.height = height
        self._seed = seed
        self._rng = random.Random(seed or random.randint(0, 2**31 - 1))

        self._pipeline = self._load_pipeline(StableDiffusionPipeline, model_path, quantize)

    @staticmethod
    def _load_pipeline(pipeline_cls: Any, model_path: str, quantize: bool) -> Any:
        load_attempts = []
        if hasattr(pipeline_cls, "from_pretrained"):
            load_attempts.append((pipeline_cls.from_pretrained, {}))
        load_attempts.append((pipeline_cls, {}))

        for loader, extra_kwargs in load_attempts:
            for keyword in ("quantize", "quantized", "quantization", "use_quantization"):
                kwargs = dict(extra_kwargs)
                kwargs[keyword] = quantize
                try:
                    return _call_with_supported_kwargs(loader, model_path, **kwargs)
                except TypeError:
                    continue
            try:
                return _call_with_supported_kwargs(loader, model_path)
            except TypeError:
                continue
        raise RuntimeError("Stable Diffusion 파이프라인을 초기화하지 못했습니다.")

    def _next_seed(self) -> int:
        if self._seed is not None:
            return self._seed
        return self._rng.randint(0, 2**31 - 1)

    def _run_pipeline(self, prompt: str, seed: int) -> Any:
        pipeline = self._pipeline
        kwargs = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": seed,
            "height": self.height,
            "width": self.width,
        }

        if hasattr(pipeline, "generate"):
            result = _call_with_supported_kwargs(pipeline.generate, **kwargs)
        else:
            result = _call_with_supported_kwargs(pipeline, **kwargs)

        if isinstance(result, (list, tuple)):
            return result[0]
        if hasattr(result, "images") and result.images:
            return result.images[0]
        return result

    def _build_prompt(self, gm_text: str, player_input: str, facts: Iterable[str]) -> str:
        fact_summary = ", ".join(facts[-6:]) if isinstance(facts, list) else ", ".join(list(facts)[-6:])
        base_prompt = [
            "detailed illustration, fantasy tabletop role playing scene",
            gm_text or "mysterious scene",
        ]
        if player_input:
            base_prompt.append(f"player action: {player_input}")
        if fact_summary:
            base_prompt.append(f"world facts: {fact_summary}")
        return ", ".join(part for part in base_prompt if part)

    def generate(self, *, gm_text: str, player_input: str, facts: List[str]) -> str:
        prompt = self._build_prompt(gm_text, player_input, facts)
        seed = self._next_seed()
        raw_image = self._run_pipeline(prompt, seed)
        return _encode_image_to_data_uri(raw_image)


def _create_game_master() -> GameMaster:
    llm = _llm_factory()
    return create_default_game_master(llm)


def _create_scene_generator() -> SceneGenerator:
    generator = _scene_generator_factory()
    if not isinstance(generator, SceneGenerator):
        raise RuntimeError("장면 생성기가 generate(gm_text, player_input, facts) 메서드를 제공해야 합니다.")
    return generator


def _generate_scene_image(
    session: Session,
    *,
    gm_text: str,
    player_input: str,
) -> str:
    generator = session.scene_generator
    if not hasattr(generator, "generate"):
        raise RuntimeError("현재 장면 생성기는 generate 메서드를 지원하지 않습니다.")

    image = generator.generate(
        gm_text=gm_text,
        player_input=player_input,
        facts=session.game_master.state.facts,
    )
    if not _is_valid_data_uri(image):
        raise ValueError("장면 생성기가 유효한 이미지 데이터 URI를 반환하지 않았습니다.")
    return image


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


def configure_mlx_stable_diffusion(
    model_path: str,
    *,
    quantize: bool = True,
    steps: int = 30,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
) -> None:
    """Configure the server to use an MLX Stable Diffusion scene generator."""

    cache: Dict[str, SceneGenerator] = {}

    def factory() -> SceneGenerator:
        if "generator" not in cache:
            cache["generator"] = MLXStableDiffusionSceneGenerator(
                model_path,
                quantize=quantize,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
            )
        return cache["generator"]

    configure_scene_generator(factory)


def _resolve_sd_model(args: argparse.Namespace) -> Optional[str]:
    """Derive the Stable Diffusion 모델 경로 or None if 비활성화."""

    if getattr(args, "sd_disable", False):
        return None

    if getattr(args, "sd_model", None):
        return args.sd_model

    env_model = os.getenv("MLX_SD_MODEL")
    if env_model:
        return env_model

    if getattr(args, "model", None):
        return DEFAULT_SD_MODEL_ID

    return None


@app.post("/api/session")
def create_session():
    identifier = uuid.uuid4().hex
    gm = _create_game_master()
    generator = _create_scene_generator()
    session = Session(
        identifier=identifier,
        game_master=gm,
        scene_generator=generator,
        scene_image=SCENE_IMAGE_PLACEHOLDER,
    )
    greeting = (
        "안녕하세요! 이곳은 소규모 판타지 마을입니다. "
        "당신의 목표는 마음이 가는 대로 모험을 펼치며 작은 영웅이 되는 것입니다."
    )
    session.record("gm", greeting)
    gm.state.add_fact(f"GM: {greeting}")
    try:
        session.scene_image = _generate_scene_image(session, gm_text=greeting, player_input="")
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
        session.scene_image = _generate_scene_image(
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
    return render_template("index.html", placeholder_image=SCENE_IMAGE_PLACEHOLDER)


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
    parser.add_argument("--sd-model", help="MLX Stable Diffusion 양자화 모델 경로")
    parser.add_argument("--sd-steps", type=int, default=30, help="Stable Diffusion 추론 스텝 수")
    parser.add_argument(
        "--sd-guidance",
        type=float,
        default=7.5,
        help="Stable Diffusion guidance scale 값",
    )
    parser.add_argument(
        "--sd-negative",
        help="Stable Diffusion에 사용할 네거티브 프롬프트",
    )
    parser.add_argument(
        "--sd-seed",
        type=int,
        help="Stable Diffusion 시드 (지정하지 않으면 매번 무작위)",
    )
    parser.add_argument("--sd-width", type=int, default=512)
    parser.add_argument("--sd-height", type=int, default=512)
    parser.add_argument(
        "--sd-no-quantize",
        action="store_true",
        help="양자화 모델을 사용하지 않도록 설정 (기본은 사용)",
    )
    parser.add_argument(
        "--sd-disable",
        action="store_true",
        help=(
            "Stable Diffusion 기반 장면 생성을 비활성화하고 SVG 키워드 생성기로 되돌립니다."
        ),
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

    resolved_sd_model = _resolve_sd_model(args)
    if resolved_sd_model:
        configure_mlx_stable_diffusion(
            resolved_sd_model,
            quantize=not args.sd_no_quantize,
            steps=args.sd_steps,
            guidance_scale=args.sd_guidance,
            negative_prompt=args.sd_negative,
            seed=args.sd_seed,
            width=args.sd_width,
            height=args.sd_height,
        )

    app.run(host=args.host, port=args.port, debug=False)
