"""Flask 서버 엔드포인트 동작을 검증하는 테스트."""

from __future__ import annotations

import pathlib
import re
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Tuple

import pytest


class _StubBaseMessage:
    def __init__(self, content: str):
        self.content = content


class _StubSystemMessage(_StubBaseMessage):
    pass


class _StubHumanMessage(_StubBaseMessage):
    pass


class _StubAIMessage(_StubBaseMessage):
    pass


def _install_message_stubs() -> None:
    core_module = types.ModuleType("langchain_core")
    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.BaseMessage = _StubBaseMessage
    messages_module.SystemMessage = _StubSystemMessage
    messages_module.HumanMessage = _StubHumanMessage
    messages_module.AIMessage = _StubAIMessage
    core_module.messages = messages_module

    schema_module = types.ModuleType("langchain.schema")
    schema_module.BaseMessage = _StubBaseMessage
    schema_module.SystemMessage = _StubSystemMessage
    schema_module.HumanMessage = _StubHumanMessage
    schema_module.AIMessage = _StubAIMessage

    sys.modules.setdefault("langchain_core", core_module)
    sys.modules.setdefault("langchain_core.messages", messages_module)
    sys.modules.setdefault("langchain.schema", schema_module)


def _install_flask_stub() -> None:
    flask_module = types.ModuleType("flask")

    @dataclass
    class _FakeResponse:
        payload: Dict[str, Any]
        status_code: int = 200

        def get_json(self) -> Dict[str, Any]:
            return dict(self.payload)

    class _RequestState:
        def __init__(self) -> None:
            self._json = None

        def set_json(self, payload: Dict[str, Any] | None) -> None:
            self._json = payload

        def get_json(self, silent: bool = False) -> Dict[str, Any] | None:
            return self._json

    request_state = _RequestState()

    def jsonify(payload: Dict[str, Any]) -> _FakeResponse:
        return _FakeResponse(dict(payload))

    def render_template(name: str, **_: Any) -> str:
        return name

    RouteFunc = Callable[..., Any]

    class Flask:
        def __init__(self, import_name: str) -> None:
            self.import_name = import_name
            self._routes: list[Tuple[str, re.Pattern[str], RouteFunc]] = []

        def _register(self, methods: Iterable[str], rule: str, func: RouteFunc) -> RouteFunc:
            pattern = re.sub(r"<([^>]+)>", r"(?P<\1>[^/]+)", rule)
            regex = re.compile(f"^{pattern}$")
            for method in methods:
                self._routes.append((method.upper(), regex, func))
            return func

        def route(self, rule: str, methods: Iterable[str]) -> Callable[[RouteFunc], RouteFunc]:
            def decorator(func: RouteFunc) -> RouteFunc:
                return self._register(methods, rule, func)

            return decorator

        def get(self, rule: str) -> Callable[[RouteFunc], RouteFunc]:
            return self.route(rule, ["GET"])

        def post(self, rule: str) -> Callable[[RouteFunc], RouteFunc]:
            return self.route(rule, ["POST"])

        def test_client(self) -> "_FakeClient":
            return _FakeClient(self)

        def run(self, *_, **__):  # pragma: no cover - not used in tests
            raise RuntimeError("Server run not supported in tests")

    class _FakeClient:
        def __init__(self, app: Flask) -> None:
            self._app = app

        def _dispatch(self, method: str, path: str, json: Dict[str, Any] | None = None):
            request_state.set_json(json)
            for registered_method, regex, func in self._app._routes:
                match = regex.match(path)
                if match and registered_method == method:
                    result = func(**match.groupdict())
                    if isinstance(result, tuple):
                        response, status = result
                    else:
                        response, status = result, 200
                    if not isinstance(response, _FakeResponse):
                        response = _FakeResponse(dict(response))
                    response.status_code = status
                    return response
            raise AssertionError(f"Route not found for {method} {path}")

        def post(self, path: str, json: Dict[str, Any] | None = None):
            return self._dispatch("POST", path, json=json)

        def get(self, path: str):
            return self._dispatch("GET", path)

    flask_module.Flask = Flask
    flask_module.jsonify = jsonify
    flask_module.render_template = render_template
    flask_module.request = request_state

    sys.modules.setdefault("flask", flask_module)


_install_message_stubs()
_install_flask_stub()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import server  # noqa: E402 (경로 조정 후 import)


@pytest.fixture(autouse=True)
def reset_server_state():
    server._sessions.clear()
    server.configure_llm(server.SimpleLocalLLM)
    yield
    server.configure_llm(server.SimpleLocalLLM)
    server._sessions.clear()


def test_create_session_returns_greeting_and_scene() -> None:
    client = server.app.test_client()

    response = client.post("/api/session")
    assert response.status_code == 200

    payload = response.get_json()
    assert isinstance(payload, dict)

    session_id = payload.get("sessionId")
    assert session_id
    assert session_id in server._sessions

    history = payload.get("history")
    assert isinstance(history, list)
    assert history[0]["role"] == "gm"
    assert "안녕하세요" in history[0]["message"]

    scene = payload.get("scene")
    assert isinstance(scene, str)
    assert "GM:" in scene

    scene_image = payload.get("sceneImage")
    assert isinstance(scene_image, str)
    assert scene_image.startswith("<svg")
    assert "<text" not in scene_image
    assert payload.get("sceneImageAlt") == history[0]["message"]


def test_send_message_updates_history_and_scene() -> None:
    client = server.app.test_client()
    session_id = client.post("/api/session").get_json()["sessionId"]

    response = client.post(
        f"/api/session/{session_id}/message",
        json={"message": "깊은 숲으로 발걸음을 옮긴다"},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["history"][-2:] == [
        {"role": "player", "message": "깊은 숲으로 발걸음을 옮긴다"},
        {"role": "gm", "message": payload["gm"]},
    ]
    assert payload["gm"].startswith("머나먼 종소리가 어렴풋이 울려 퍼진다.")
    assert payload["sceneImage"].startswith("<svg")
    assert "<text" not in payload["sceneImage"]
    assert payload["sceneImageAlt"] == payload["gm"]


def test_can_inject_custom_llm_via_configure_llm() -> None:
    class _EchoLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages):  # type: ignore[override]
            self.calls += 1
            return f"ECHO {self.calls}: {messages[-1].content}"

        def draw_scene(self, *, gm_text: str, player_input: str, facts: list[str]) -> str:
            return (
                "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'>"
                f"<text x='5' y='5'>{gm_text or player_input}</text></svg>"
            )

    server.configure_llm(lambda: _EchoLLM())
    client = server.app.test_client()
    session_id = client.post("/api/session").get_json()["sessionId"]

    response = client.post(
        f"/api/session/{session_id}/message",
        json={"message": "테스트"},
    )

    payload = response.get_json()
    assert payload["gm"].startswith("ECHO 1:")
    assert "Player: 테스트" in payload["scene"]
    assert "GM:" in payload["scene"]
    assert payload["sceneImage"].startswith("<svg")


def test_scene_image_updates_from_gm_description() -> None:
    class _CastleLLM:
        def invoke(self, _messages):  # type: ignore[override]
            return "거대한 성벽과 궁전이 시야를 가득 채운다."

        def draw_scene(self, *, gm_text: str, player_input: str, facts: list[str]) -> str:
            return "<svg xmlns='http://www.w3.org/2000/svg'><text>castle</text></svg>"

    server.configure_llm(lambda: _CastleLLM())
    client = server.app.test_client()
    session_id = client.post("/api/session").get_json()["sessionId"]

    response = client.post(
        f"/api/session/{session_id}/message",
        json={"message": "앞에 보이는 건물에 다가간다"},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["sceneImage"] == "<svg xmlns='http://www.w3.org/2000/svg'><text>castle</text></svg>"
    assert payload["sceneImageAlt"].startswith("거대한 성벽")


def test_send_message_validates_input() -> None:
    client = server.app.test_client()
    session_id = client.post("/api/session").get_json()["sessionId"]

    response = client.post(f"/api/session/{session_id}/message", json={"message": "  "})
    assert response.status_code == 400

    payload = response.get_json()
    assert payload == {"error": "메시지를 입력해 주세요."}


def test_send_message_unknown_session_returns_404() -> None:
    client = server.app.test_client()

    response = client.post(
        "/api/session/does-not-exist/message",
        json={"message": "무언가 한다"},
    )
    assert response.status_code == 404
    assert response.get_json() == {"error": "세션을 찾을 수 없습니다."}


def test_configure_lmstudio_model_reuses_single_storyteller(monkeypatch) -> None:
    created: list[tuple[str, Dict[str, Any]]] = []

    class _StubStoryteller:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            created.append((model_name, kwargs))

        def invoke(self, _messages):  # type: ignore[override]
            return "STUB RESPONSE"

        def draw_scene(self, *, gm_text: str, player_input: str, facts: list[str]) -> str:
            return "<svg xmlns='http://www.w3.org/2000/svg'><text>stub</text></svg>"

    monkeypatch.setattr(server, "LMStudioStoryteller", _StubStoryteller)

    server.configure_lmstudio_model(
        "stub-model",
        temperature=0.5,
        max_tokens=256,
        api_base="http://example.com/v1",
        api_key="secret-token",
    )

    client = server.app.test_client()
    session_one = client.post("/api/session").get_json()["sessionId"]
    response_one = client.post(
        f"/api/session/{session_one}/message",
        json={"message": "주위를 살펴본다"},
    )
    assert response_one.status_code == 200
    assert response_one.get_json()["gm"] == "STUB RESPONSE"

    session_two = client.post("/api/session").get_json()["sessionId"]
    response_two = client.post(
        f"/api/session/{session_two}/message",
        json={"message": "다시 시도"},
    )
    assert response_two.status_code == 200

    assert response_two.get_json()["sceneImage"].startswith("<svg")

    assert len(created) == 1
    model_name, kwargs = created[0]
    assert model_name == "stub-model"
    assert kwargs == {
        "temperature": 0.5,
        "max_tokens": 256,
        "api_base": "http://example.com/v1",
        "api_key": "secret-token",
    }
