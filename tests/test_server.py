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
def clear_sessions():
    server._sessions.clear()
    yield
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


def test_send_message_updates_history_and_scene() -> None:
    client = server.app.test_client()
    session_id = client.post("/api/session").get_json()["sessionId"]

    response = client.post(
        f"/api/session/{session_id}/message",
        json={"message": "주변을 살핀다"},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["history"][-2:] == [
        {"role": "player", "message": "주변을 살핀다"},
        {"role": "gm", "message": payload["gm"]},
    ]
    assert payload["gm"].startswith("머나먼 종소리가 어렴풋이 울려 퍼진다.")
    assert "Player: 주변을 살핀다" in payload["scene"]
    assert "GM:" in payload["scene"]


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
