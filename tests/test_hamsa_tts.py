from __future__ import annotations

import json

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, APIError
from livekit.plugins import hamsa


def _text_message(payload: dict) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(payload), "")


def _binary_message(data: bytes) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, data, "")


class FakeWebSocket:
    def __init__(self, messages: list[aiohttp.WSMessage]) -> None:
        self._messages = messages
        self.sent_text: list[str] = []
        self.close_code = 1000
        self.closed = False

    async def send_str(self, data: str) -> None:
        self.sent_text.append(data)

    async def receive(self) -> aiohttp.WSMessage:
        if self._messages:
            return self._messages.pop(0)

        self.closed = True
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, "")

    async def close(self) -> None:
        self.closed = True

    def exception(self) -> None:
        return None


class FakeSession:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self._websocket = websocket
        self.ws_urls: list[str] = []

    async def ws_connect(self, url: str, **kwargs: object) -> FakeWebSocket:
        self.ws_urls.append(url)
        return self._websocket


def _pcm_audio(frame_size: int = 4800) -> bytes:
    return b"\x00\x00" * frame_size


def test_hamsa_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HAMSA_API_KEY", raising=False)

    with pytest.raises(ValueError, match="HAMSA API key is required"):
        hamsa.TTS(speaker="Amjad")


@pytest.mark.asyncio
async def test_hamsa_stream_sends_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAMSA_API_KEY", "test-key")
    ws = FakeWebSocket(
        [
            _text_message({"type": "ack", "payload": {"message": "ok"}}),
            _binary_message(_pcm_audio()),
            _text_message({"type": "end", "payload": {"message": "done"}}),
        ]
    )
    session = FakeSession(ws)
    tts = hamsa.TTS(
        speaker="Amjad",
        dialect="pls",
        language_id="ar",
        http_session=session,  # type: ignore[arg-type]
    )

    try:
        async with tts.stream() as stream:
            stream.push_text("hello ")
            stream.push_text("world")
            stream.end_input()
            events = [event async for event in stream]

        assert events
        assert events[-1].is_final
        assert "api_key=test-key" in session.ws_urls[0]

        request = json.loads(ws.sent_text[0])
        assert request == {
            "type": "tts",
            "payload": {
                "text": "hello world",
                "speaker": "Amjad",
                "dialect": "pls",
                "languageId": "ar",
                "mulaw": False,
            },
        }
    finally:
        await tts.aclose()


@pytest.mark.asyncio
async def test_hamsa_synthesize_uses_stream_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAMSA_API_KEY", "test-key")
    ws = FakeWebSocket(
        [
            _text_message({"type": "ack", "payload": {"message": "ok"}}),
            _binary_message(_pcm_audio()),
            _text_message({"type": "end", "payload": {"message": "done"}}),
        ]
    )
    session = FakeSession(ws)
    tts = hamsa.TTS(speaker="Amjad", http_session=session)  # type: ignore[arg-type]

    try:
        async with tts.synthesize("hello from hamsa") as stream:
            frame = await stream.collect()

        assert frame.sample_rate == tts.sample_rate
        assert frame.num_channels == tts.num_channels
        request = json.loads(ws.sent_text[0])
        assert request["payload"]["text"] == "hello from hamsa"
    finally:
        await tts.aclose()


@pytest.mark.asyncio
async def test_hamsa_stream_raises_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAMSA_API_KEY", "test-key")
    ws = FakeWebSocket(
        [
            _text_message(
                {"type": "error", "payload": {"message": "Invalid payload for message type: tts"}}
            ),
        ]
    )
    session = FakeSession(ws)
    tts = hamsa.TTS(speaker="Amjad", http_session=session)  # type: ignore[arg-type]

    try:
        async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0)) as stream:
            stream.push_text("hello")
            stream.end_input()
            with pytest.raises(APIError, match="Invalid payload for message type: tts"):
                async for _ in stream:
                    pass
    finally:
        await tts.aclose()
