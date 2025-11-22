from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest

import json
import threading
from http.client import HTTPConnection

import pytest

from server import GameMasterError, TRPGHTTPServer, create_app
from trpg import SceneImageResult


class DummyGameMaster:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def respond(self, message: str) -> str:
        self.messages.append(message)
        return f"응답: {message}"

    def render_scene(self, *, width: int = 60) -> str:  # pylint: disable=unused-argument
        if not self.messages:
            return "(빈 장면)"
        return "\n".join(self.messages)


def _factory() -> DummyGameMaster:
    return DummyGameMaster()


class FailingGameMaster(DummyGameMaster):
    def respond(self, message: str) -> str:  # noqa: D401 - 테스트용 더미
        raise RuntimeError("No models loaded")


class VisualGameMaster(DummyGameMaster):
    def render_scene_image(self):  # pragma: no cover - 단순 데이터 반환
        return SceneImageResult(
            prompt="밤의 성",
            data_url="data:image/png;base64,abc123",
            data_urls=["data:image/png;base64,abc123", "data:image/png;base64,def456"],
            error=None,
        )


class BrokenVisualGameMaster(DummyGameMaster):
    def render_scene_image(self):  # pragma: no cover - 예외 경로 확인
        raise RuntimeError("렌더링 실패")


def test_app_create_session() -> None:
    app = create_app(factory=_factory)

    payload = app.create_session()

    assert "session_id" in payload
    assert payload["scene"] == "(빈 장면)"


def test_app_send_message_updates_scene() -> None:
    app = create_app(factory=_factory)
    session = app.create_session()

    payload = app.send_message(session["session_id"], "문을 연다")

    assert payload["response"] == "응답: 문을 연다"
    assert payload["scene"] == "문을 연다"


def test_app_includes_scene_image_when_available() -> None:
    app = create_app(factory=VisualGameMaster)

    payload = app.create_session()

    assert payload["scene"].startswith("(빈 장면)")
    assert payload["scene_image"].startswith("data:image/png;base64,")
    assert payload["scene_images"][0] == payload["scene_image"]
    assert len(payload["scene_images"]) == 2
    assert payload["scene_prompt"] == "밤의 성"


def test_app_handles_scene_image_errors() -> None:
    app = create_app(factory=BrokenVisualGameMaster)

    payload = app.create_session()

    assert "scene_image_error" in payload
    assert "렌더링 실패" in payload["scene_image_error"]


def test_app_unknown_session_raises() -> None:
    app = create_app(factory=_factory)

    try:
        app.send_message("missing", "테스트")
    except KeyError as exc:
        assert str(exc) in {"'세션을 찾을 수 없습니다.'", "세션을 찾을 수 없습니다."}
    else:  # pragma: no cover - 방어적 코드
        raise AssertionError("KeyError가 발생해야 합니다.")


def test_app_send_message_wraps_errors() -> None:
    app = create_app(factory=FailingGameMaster)
    session = app.create_session()

    with pytest.raises(GameMasterError) as excinfo:
        app.send_message(session["session_id"], "테스트")

    assert "게임 마스터가 응답을 생성하지 못했습니다." in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_http_endpoints() -> None:
    app = create_app(factory=_factory)
    server = TRPGHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    conn = HTTPConnection(host, port)

    try:
        conn.request("POST", "/api/session")
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        session_id = body["session_id"]

        conn.request(
            "POST",
            f"/api/session/{session_id}/message",
            body=json.dumps({"message": "문을 닫는다"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        message_body = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert message_body["response"] == "응답: 문을 닫는다"
        assert "문을 닫는다" in message_body["scene"]
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_http_returns_bad_gateway_on_game_master_error() -> None:
    app = create_app(factory=FailingGameMaster)
    server = TRPGHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    conn = HTTPConnection(host, port)

    try:
        conn.request("POST", "/api/session")
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        session_id = body["session_id"]

        conn.request(
            "POST",
            f"/api/session/{session_id}/message",
            body=json.dumps({"message": "문을 연다"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 502
        assert payload["detail"].startswith("게임 마스터가 응답을 생성하지 못했습니다.")
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
