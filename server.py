"""표준 라이브러리 기반 TRPG 웹 서버."""

from __future__ import annotations

import json
import os
import textwrap
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlparse
from uuid import uuid4

from main import build_game_master
from trpg import GameMaster

GameMasterFactory = Callable[[], GameMaster]


@dataclass
class WebResponse:
    status: int
    headers: Dict[str, str]
    body: bytes


class SessionStore:
    """게임 마스터 인스턴스를 관리하는 스레드 안전 저장소."""

    def __init__(self, factory: GameMasterFactory) -> None:
        self._factory = factory
        self._sessions: Dict[str, GameMaster] = {}
        self._lock = threading.Lock()

    def create(self) -> Tuple[str, GameMaster]:
        with self._lock:
            session_id = uuid4().hex
            game_master = self._factory()
            self._sessions[session_id] = game_master
            return session_id, game_master

    def get(self, session_id: str) -> GameMaster:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("세션을 찾을 수 없습니다.")
            return self._sessions[session_id]


class GameMasterError(RuntimeError):
    """게임 마스터가 응답 생성에 실패했을 때 사용되는 예외."""


class WebApp:
    """웹 요청을 처리하는 순수 파이썬 애플리케이션."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def create_session(self) -> Dict[str, str]:
        session_id, game_master = self._store.create()
        payload = {"session_id": session_id}
        payload.update(self._scene_payload(game_master))
        return payload

    def send_message(self, session_id: str, message: str) -> Dict[str, str]:
        message = (message or "").strip()
        if not message:
            raise ValueError("메시지는 비어 있을 수 없습니다.")
        game_master = self._store.get(session_id)
        try:
            response = game_master.respond(message)
        except Exception as exc:
            raise GameMasterError("게임 마스터가 응답을 생성하지 못했습니다. 모델 구성을 확인하세요.") from exc
        payload = {"response": response}
        payload.update(self._scene_payload(game_master))
        return payload

    @staticmethod
    def _scene_payload(game_master: GameMaster) -> Dict[str, str]:
        payload: Dict[str, str] = {"scene": game_master.render_scene()}
        render_scene_image = getattr(game_master, "render_scene_image", None)
        if callable(render_scene_image):
            try:
                image_result = render_scene_image()
            except Exception as exc:  # pragma: no cover - 런타임 방어
                payload["scene_image_error"] = f"장면 이미지를 생성하지 못했습니다: {exc}"
            else:
                if image_result:
                    if image_result.data_urls:
                        payload["scene_images"] = image_result.data_urls
                    if image_result.data_url:
                        payload["scene_image"] = image_result.data_url
                    if image_result.prompt:
                        payload["scene_prompt"] = image_result.prompt
                    if image_result.error:
                        payload["scene_image_error"] = image_result.error
        return payload

    @staticmethod
    def index_html() -> str:
        return build_index_html()


def default_factory() -> GameMaster:
    model = os.getenv("TRPG_MODEL", "gpt-3.5-turbo")
    temperature = float(os.getenv("TRPG_TEMPERATURE", "0.7"))
    api_base: Optional[str] = os.getenv("TRPG_API_BASE")
    api_key: Optional[str] = os.getenv("TRPG_API_KEY")
    return build_game_master(
        model=model,
        temperature=temperature,
        api_base=api_base,
        api_key=api_key,
    )


def create_app(factory: Optional[GameMasterFactory] = None) -> WebApp:
    return WebApp(SessionStore(factory or default_factory))


class TRPGRequestHandler(BaseHTTPRequestHandler):
    server_version = "TRPGServer/1.0"

    def _dispatch(self) -> WebResponse:
        parsed = urlparse(self.path)
        app: WebApp = self.server.app  # type: ignore[attr-defined]

        if self.command == "GET" and parsed.path == "/":
            body = app.index_html().encode("utf-8")
            return WebResponse(
                status=HTTPStatus.OK,
                headers={"Content-Type": "text/html; charset=utf-8"},
                body=body,
            )

        if self.command == "POST" and parsed.path == "/api/session":
            payload = app.create_session()
            return _json_response(payload)

        if self.command == "POST" and parsed.path.startswith("/api/session/"):
            try:
                _, _, _, session_id, action = parsed.path.split("/", 4)
            except ValueError:
                return _json_error(HTTPStatus.NOT_FOUND, "세션을 찾을 수 없습니다.")
            if action != "message":
                return _json_error(HTTPStatus.NOT_FOUND, "지원하지 않는 경로입니다.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return _json_error(HTTPStatus.BAD_REQUEST, "잘못된 요청 본문 길이입니다.")
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return _json_error(HTTPStatus.BAD_REQUEST, "JSON 파싱에 실패했습니다.")
            message = data.get("message", "")
            try:
                payload = app.send_message(session_id, message)
            except KeyError as exc:
                return _json_error(HTTPStatus.NOT_FOUND, str(exc))
            except ValueError as exc:
                return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
            except GameMasterError as exc:
                self.log_error("Game master failure: %s", exc)
                return _json_error(HTTPStatus.BAD_GATEWAY, str(exc))
            return _json_response(payload)

        return _json_error(HTTPStatus.NOT_FOUND, "지원하지 않는 경로입니다.")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        response = self._dispatch()
        self._write_response(response)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        response = self._dispatch()
        self._write_response(response)

    def _write_response(self, response: WebResponse) -> None:
        self.send_response(response.status)
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - signature fixed
        return  # 서버 테스트 시 출력 억제


class TRPGHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: Tuple[str, int], app: WebApp) -> None:
        super().__init__(server_address, TRPGRequestHandler)
        self.app = app


def build_index_html() -> str:
    return textwrap.dedent(
        """
        <!doctype html>
        <html lang="ko">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>LangChain TRPG</title>
            <style>
                :root {
                    color-scheme: light dark;
                    font-family: 'Noto Sans KR', system-ui, sans-serif;
                    background: #0f172a;
                    color: #f8fafc;
                }
                body {
                    margin: 0;
                    display: grid;
                    place-items: center;
                    min-height: 100vh;
                    padding: 1.5rem;
                }
                .card {
                    width: min(720px, 100%);
                    background: rgba(15, 23, 42, 0.8);
                    border: 1px solid rgba(148, 163, 184, 0.3);
                    border-radius: 16px;
                    padding: 2rem;
                    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.45);
                }
                h1 {
                    margin-top: 0;
                    font-size: 2rem;
                    letter-spacing: -0.03em;
                }
                .panel {
                    background: rgba(15, 23, 42, 0.6);
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    border-radius: 12px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                }
                #scene-text {
                    white-space: pre-wrap;
                }
                #log {
                    height: 320px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                }
                form {
                    display: flex;
                    gap: 0.75rem;
                }
                input[type="text"] {
                    flex: 1;
                    padding: 0.75rem 1rem;
                    border-radius: 999px;
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    background: rgba(15, 23, 42, 0.8);
                    color: inherit;
                    font-size: 1rem;
                }
                button {
                    padding: 0.75rem 1.5rem;
                    border-radius: 999px;
                    border: none;
                    background: linear-gradient(135deg, #38bdf8, #6366f1);
                    color: white;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.15s ease;
                }
                button:disabled {
                    opacity: 0.6;
                    cursor: wait;
                }
                button:not(:disabled):hover {
                    transform: translateY(-1px);
                }
                .scene-visual {
                    border: 1px solid rgba(148, 163, 184, 0.25);
                    border-radius: 12px;
                    padding: 0.75rem;
                    background: rgba(15, 23, 42, 0.5);
                }
                .scene-gallery {
                    display: grid;
                    gap: 0.75rem;
                }
                .scene-frame img {
                    display: block;
                    width: 100%;
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
                }
                .scene-variations {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 0.5rem;
                }
                .scene-variations[data-empty="true"] {
                    display: none;
                }
                .scene-variation {
                    opacity: 0;
                    transform: translateY(10px);
                    transition: opacity 0.3s ease, transform 0.3s ease;
                }
                .scene-variation img {
                    width: 100%;
                    border-radius: 10px;
                    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
                    border: 1px solid rgba(148, 163, 184, 0.25);
                }
                .scene-variation.pop-in {
                    opacity: 1;
                    transform: translateY(0);
                    animation: popIn 0.45s ease forwards;
                }
                @keyframes popIn {
                    0% { opacity: 0; transform: translateY(12px) scale(0.98); }
                    100% { opacity: 1; transform: translateY(0) scale(1); }
                }
                .eyebrow {
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    font-size: 0.75rem;
                    color: rgba(148, 163, 184, 0.85);
                    margin: 0 0 0.35rem;
                }
                .scene-prompt {
                    font-size: 0.9rem;
                    color: rgba(226, 232, 240, 0.85);
                    margin: 0.5rem 0 0;
                    white-space: pre-wrap;
                }
                .scene-error {
                    color: #fca5a5;
                    margin: 0.25rem 0 0.75rem;
                }
                footer {
                    margin-top: 1.5rem;
                    font-size: 0.85rem;
                    color: rgba(226, 232, 240, 0.7);
                }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>LangChain TRPG</h1>
                <div id="scene-visual" class="scene-visual" hidden>
                    <p class="eyebrow">Stable Diffusion (MLX)</p>
                    <div id="scene-gallery" class="scene-gallery" hidden>
                        <div class="scene-frame primary-frame">
                            <img id="scene-image" alt="Stable Diffusion으로 생성된 장면" loading="lazy">
                        </div>
                        <div id="scene-variations" class="scene-variations" data-empty="true"></div>
                    </div>
                    <p id="scene-prompt" class="scene-prompt"></p>
                </div>
                <div id="scene-error" class="scene-error" hidden></div>
                <div id="scene-text" class="panel">(장면을 준비하는 중...)</div>
                <div id="log" class="panel">새 세션을 준비하는 중...</div>
                <form id="input-form">
                    <input id="message" type="text" placeholder="행동을 입력하세요" autocomplete="off">
                    <button type="submit">보내기</button>
                </form>
                <footer>Powered by LangChain • 한국어로 이야기하세요!</footer>
            </div>
            <script>
                const log = document.getElementById('log');
                const form = document.getElementById('input-form');
                const input = document.getElementById('message');
                const sceneText = document.getElementById('scene-text');
                const sceneVisual = document.getElementById('scene-visual');
                const sceneGallery = document.getElementById('scene-gallery');
                const sceneImage = document.getElementById('scene-image');
                const sceneVariations = document.getElementById('scene-variations');
                const scenePrompt = document.getElementById('scene-prompt');
                const sceneError = document.getElementById('scene-error');
                let variationTimers = [];
                let sessionId = null;

                async function createSession() {
                    const response = await fetch('/api/session', { method: 'POST' });
                    if (!response.ok) {
                        log.textContent = '세션 생성에 실패했습니다. 서버 로그를 확인하세요.';
                        form.style.display = 'none';
                        return;
                    }
                    const data = await response.json();
                    sessionId = data.session_id;
                    log.textContent = '새 세션이 시작되었습니다.';
                    renderScene(data);
                }

                function renderVariations(sources) {
                    variationTimers.forEach(clearTimeout);
                    variationTimers = [];
                    sceneVariations.innerHTML = '';
                    if (!sources.length) {
                        sceneVariations.dataset.empty = 'true';
                        return;
                    }
                    sceneVariations.dataset.empty = 'false';
                    sources.forEach((src, index) => {
                        const timer = setTimeout(() => {
                            const frame = document.createElement('div');
                            frame.className = 'scene-variation';
                            const img = document.createElement('img');
                            img.loading = 'lazy';
                            img.src = src;
                            img.alt = `Stable Diffusion 변주 ${index + 2}`;
                            img.addEventListener('load', () => {
                                requestAnimationFrame(() => {
                                    frame.classList.add('pop-in');
                                });
                            });
                            frame.appendChild(img);
                            sceneVariations.appendChild(frame);
                        }, index * 320);
                        variationTimers.push(timer);
                    });
                }

                function renderScene(payload) {
                    if (payload.scene) {
                        sceneText.textContent = payload.scene;
                    }

                    const images = Array.isArray(payload.scene_images)
                        ? payload.scene_images
                        : (payload.scene_image ? [payload.scene_image] : []);

                    if (images.length) {
                        sceneImage.src = images[0];
                        sceneVisual.hidden = false;
                        sceneGallery.hidden = false;
                        renderVariations(images.slice(1));
                        if (payload.scene_prompt) {
                            scenePrompt.textContent = `프롬프트: ${payload.scene_prompt}`;
                            scenePrompt.style.display = 'block';
                        } else {
                            scenePrompt.textContent = '';
                            scenePrompt.style.display = 'none';
                        }
                    } else {
                        sceneVisual.hidden = true;
                        sceneGallery.hidden = true;
                        sceneVariations.innerHTML = '';
                        sceneVariations.dataset.empty = 'true';
                        sceneImage.removeAttribute('src');
                        scenePrompt.textContent = '';
                        scenePrompt.style.display = 'none';
                    }

                    if (payload.scene_image_error) {
                        sceneError.textContent = payload.scene_image_error;
                        sceneError.hidden = false;
                    } else {
                        sceneError.textContent = '';
                        sceneError.hidden = true;
                    }
                }

                function appendResponse(message, response) {
                    log.textContent += `\n\n플레이어: ${message}\nGM: ${response}`;
                    log.scrollTop = log.scrollHeight;
                }

                form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const message = input.value.trim();
                    if (!message || !sessionId) {
                        return;
                    }
                    form.querySelector('button').disabled = true;
                    try {
                        const response = await fetch(`/api/session/${sessionId}/message`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message })
                        });
                        if (!response.ok) {
                            appendResponse(message, '응답을 가져오지 못했습니다.');
                            return;
                        }
                        const data = await response.json();
                        appendResponse(message, data.response);
                        renderScene(data);
                        input.value = '';
                        input.focus();
                    } finally {
                        form.querySelector('button').disabled = false;
                    }
                });

                createSession();
            </script>
        </body>
        </html>
        """
    ).strip()


def _json_response(payload: Dict[str, str]) -> WebResponse:
    body = json.dumps(payload).encode("utf-8")
    return WebResponse(
        status=HTTPStatus.OK,
        headers={"Content-Type": "application/json; charset=utf-8"},
        body=body,
    )


def _json_error(status: HTTPStatus, message: str) -> WebResponse:
    body = json.dumps({"detail": message}).encode("utf-8")
    return WebResponse(
        status=status,
        headers={"Content-Type": "application/json; charset=utf-8"},
        body=body,
    )


def run(host: str = "127.0.0.1", port: int = 8000, factory: Optional[GameMasterFactory] = None) -> None:
    app = create_app(factory)
    server = TRPGHTTPServer((host, port), app)
    print(f"🌐 TRPG 서버가 http://{host}:{port} 에서 실행 중입니다. (Ctrl+C로 종료)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - 인터랙티브 종료
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover - 수동 실행 진입점
    run()
