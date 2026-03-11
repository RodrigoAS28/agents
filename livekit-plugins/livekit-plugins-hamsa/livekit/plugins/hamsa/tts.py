# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass, replace
from urllib.parse import urlencode

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger
from .version import __version__

DEFAULT_BASE_URL = "https://api.tryhamsa.com"
# Websocket API example uses "modern"; HTTP API uses country codes (pls, leb, egy, etc.)
DEFAULT_DIALECT = "modern"
DEFAULT_LANGUAGE_ID = "ar"
DEFAULT_PCM_SAMPLE_RATE = 16000
DEFAULT_MULAW_SAMPLE_RATE = 8000


@dataclass
class _TTSOptions:
    speaker: str
    dialect: str
    language_id: str
    mulaw: bool
    api_key: str
    base_url: str
    sample_rate: int
    mime_type: str

    def get_ws_url(self) -> str:
        query = urlencode({"api_key": self.api_key})
        return f"{self.base_url.replace('http', 'ws', 1)}/v1/realtime/ws?{query}"

    def get_ws_headers(self) -> dict[str, str]:
        """Authorization header matching HTTP API (Token <key>) for TTS auth."""
        return {"Authorization": f"Token {self.api_key}"}


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        speaker: str,
        api_key: str | None = None,
        dialect: str = DEFAULT_DIALECT,
        language_id: str = DEFAULT_LANGUAGE_ID,
        mulaw: bool = False,
        sample_rate: int | None = None,
        mime_type: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of HAMSA TTS.

        Args:
            speaker: Built-in speaker name or a UUID for a cloned voice.
            api_key: HAMSA API key. Defaults to the `HAMSA_API_KEY` environment variable.
            dialect: HAMSA dialect identifier. Websocket docs use "modern"; you can also try
                country codes (pls, leb, egy, etc.) if supported.
            language_id: HAMSA language code.
            mulaw: Whether to request mu-law audio from HAMSA.
            sample_rate: Expected output sample rate. Defaults to 16kHz PCM or 8kHz mu-law.
            mime_type: MIME type for streamed audio. Defaults to `audio/pcm` or `audio/basic`.
            base_url: Base URL for the HAMSA API.
            http_session: Existing aiohttp session to reuse.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate
            if sample_rate is not None
            else (DEFAULT_MULAW_SAMPLE_RATE if mulaw else DEFAULT_PCM_SAMPLE_RATE),
            num_channels=1,
        )

        hamsa_api_key = api_key or os.environ.get("HAMSA_API_KEY")
        if not hamsa_api_key:
            raise ValueError(
                "HAMSA API key is required, either as argument or set HAMSA_API_KEY"
                " environment variable"
            )

        resolved_sample_rate = (
            sample_rate
            if sample_rate is not None
            else (DEFAULT_MULAW_SAMPLE_RATE if mulaw else DEFAULT_PCM_SAMPLE_RATE)
        )
        resolved_mime_type = mime_type or ("audio/basic" if mulaw else "audio/pcm")

        self._opts = _TTSOptions(
            speaker=speaker,
            dialect=dialect,
            language_id=language_id,
            mulaw=mulaw,
            api_key=hamsa_api_key,
            base_url=base_url.rstrip("/"),
            sample_rate=resolved_sample_rate,
            mime_type=resolved_mime_type,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        logger.info(
            "livekit-plugins-hamsa version %s (speaker=%s, dialect=%s)",
            __version__,
            speaker,
            dialect,
        )

    @property
    def model(self) -> str:
        return "realtime-ws-tts"

    @property
    def provider(self) -> str:
        return "HAMSA"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._ensure_session()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments: list[str] = []

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=self._opts.mime_type,
            stream=True,
        )

        if self._segments:
            for segment_text in self._segments:
                await self._run_segment(segment_text, output_emitter)
            return

        text_parts: list[str] = []
        async for input_item in self._input_ch:
            if isinstance(input_item, str):
                text_parts.append(input_item)
                continue

            segment_text = "".join(text_parts).strip()
            text_parts.clear()
            if segment_text:
                self._segments.append(segment_text)
                await self._run_segment(segment_text, output_emitter)

        remaining_text = "".join(text_parts).strip()
        if remaining_text:
            self._segments.append(remaining_text)
            await self._run_segment(remaining_text, output_emitter)

    async def _run_segment(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await asyncio.wait_for(
                self._tts._ensure_session().ws_connect(
                    self._opts.get_ws_url(),
                    headers=self._opts.get_ws_headers(),
                ),
                timeout=self._conn_options.timeout,
            )
            self._mark_started()
            request_payload = {
                "type": "tts",
                "payload": {
                    "text": text,
                    "speaker": self._opts.speaker,
                    "dialect": self._opts.dialect,
                    "languageId": self._opts.language_id,
                    "mulaw": self._opts.mulaw,
                },
            }
            logger.debug(
                "HAMSA TTS request: payload=%s text_len=%d text_preview=%s",
                request_payload,
                len(text),
                repr(text[:200]) + ("..." if len(text) > 200 else ""),
            )
            await asyncio.wait_for(
                ws.send_str(json.dumps(request_payload)),
                timeout=self._conn_options.timeout,
            )

            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=self._conn_options.timeout)

                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError as e:
                        raise APIError(f"invalid HAMSA websocket payload: {msg.data}") from e

                    msg_type = payload.get("type")
                    if msg_type == "ack":
                        logger.debug("HAMSA TTS acknowledged stream")
                        continue
                    if msg_type == "info":
                        logger.debug(
                            "HAMSA TTS info: %s",
                            payload.get("payload", {}).get("message", payload),
                        )
                        continue
                    if msg_type == "end":
                        output_emitter.end_segment()
                        break
                    if msg_type == "error":
                        err_payload = payload.get("payload") or {}
                        error_message = err_payload.get("message") or "unknown error"
                        logger.warning(
                            "HAMSA TTS error: %s (full payload: %s) [request: speaker=%s dialect=%s languageId=%s text_len=%d]",
                            error_message,
                            payload,
                            self._opts.speaker,
                            self._opts.dialect,
                            self._opts.language_id,
                            len(text),
                        )
                        raise APIError(f"HAMSA returned error: {error_message}")

                    logger.debug("Ignoring unexpected HAMSA websocket message: %s", payload)
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "HAMSA websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError() from ws.exception()

                logger.debug("Ignoring unexpected HAMSA websocket frame type: %s", msg.type)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None and not ws.closed:
                await ws.close()
