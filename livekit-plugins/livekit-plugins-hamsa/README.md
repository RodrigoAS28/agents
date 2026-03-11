# HAMSA plugin for LiveKit Agents

Support for streaming speech synthesis with the [HAMSA](https://docs.tryhamsa.com/websocket/websocket-tts) websocket API.

## Installation

```bash
pip install livekit-plugins-hamsa
```

## Pre-requisites

You'll need a HAMSA API key. It can be set as an environment variable: `HAMSA_API_KEY`

## Usage

```python
from livekit.plugins import hamsa

tts = hamsa.TTS(
    speaker="Amjad",
    dialect="modern",  # or country code: pls, leb, egy, etc.
    language_id="ar",
)
```

If you need to override the output assumptions, you can also set `sample_rate`, `mime_type`,
or `mulaw=True`. The plugin defaults to `16_000` Hz PCM or `8_000` Hz mu-law and keeps those
settings overridable because HAMSA's websocket docs do not currently publish the exact stream
format details.

## Troubleshooting

If you see **"TTS stream error: aborted"** (server accepts then aborts before audio):

- Confirm the **speaker** name in the [HAMSA voice dashboard](https://docs.tryhamsa.com/websocket/websocket-tts) (e.g. pre-built names or your cloned voice ID). Try a documented name like `Amjad` or `Salma` to isolate speaker issues.
- Try **dialect="modern"** once; some setups prefer it over country codes (`ksa`, `pls`, etc.).
- For **custom/cloned voices**, preload the voice via the [Preload endpoint](https://docs.tryhamsa.com/websocket/websocket-tts) before using it on the websocket.
