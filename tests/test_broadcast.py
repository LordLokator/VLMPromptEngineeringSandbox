import asyncio
import threading

import web_engine


class FakeSocket:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    async def send_text(self, text):
        if self.fail:
            raise RuntimeError("closed")
        self.sent.append(text)


def test_broadcast_prunes_dead_socket():
    good, dead = FakeSocket(), FakeSocket(fail=True)
    web_engine.active_connections[:] = [good, dead]
    asyncio.run(web_engine.broadcast({"type": "system", "text": "hi"}))
    assert len(good.sent) == 1
    assert web_engine.active_connections == [good]
    web_engine.active_connections.clear()


def test_broadcast_sync_from_thread_reaches_clients(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(web_engine.app.state, "prompt", None, raising=False)
    monkeypatch.setattr(web_engine.app.state, "settings", None, raising=False)
    web_engine.active_connections.clear()
    with TestClient(web_engine.app) as client:
        with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
            t = threading.Thread(target=web_engine.broadcast_sync, args=({"type": "vlm", "text": "x"},))
            t.start()
            t.join(5)
            assert a.receive_json()["text"] == "x"
            assert b.receive_json()["text"] == "x"
    web_engine.active_connections.clear()


def test_broadcast_sync_without_loop_is_noop(monkeypatch):
    monkeypatch.setattr(web_engine.app.state, "loop", None, raising=False)
    web_engine.broadcast_sync({"type": "vlm", "text": "x"})
