# Erebus

Erebus is a personal AI companion: a browser interface with an animated 3D face,
a persistent character, a structured long-term memory, local speech recognition,
camera-based perception and face recognition, and text-to-speech — all running
on commodity hardware with a single paid API key.

It was built as a personal project (and documented as a Master's thesis) around
one specific idea: a frontier conversational model is capable enough to be a
genuinely useful standing companion, but the way most people reach one — a
stateless browser tab that forgets everything the moment you close it — throws
all of that away. Erebus is an attempt to close that gap.

## What's here

- **`erebus.html`** — the whole interface: a single-page app with an animated,
  low-poly 3D face (Three.js), personality sliders, camera/vision controls,
  face recognition, a memory graph viewer, and the client-side conversation
  logic (streaming replies, tool use, text-to-speech playback, jaw animation
  driven by the actual audio waveform).
- **`server.py`** — a small, dependency-free local Python server. It proxies
  requests to the Anthropic and ElevenLabs APIs (so no API key is ever exposed
  to the open internet), and serves the memory/knowledge-vault API.
- **`vault.py`** — a structured knowledge graph (entities + relationships)
  that Erebus draws on for memory, instead of a flat list of facts.
- **`whisper_mic.py`** — an optional, fully local and offline speech-to-text
  service (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)),
  used instead of the browser's built-in speech recognition (which sends audio
  to Google and doesn't exist on a Raspberry Pi).
- **`START_EREBUS.bat`** — a convenience launcher for Windows.

## Setup

1. **Install Python 3** (3.10+ recommended).
2. **Get your API keys:**
   - An [Anthropic API key](https://console.anthropic.com/) (required — this is
     the model Erebus talks with).
   - An [ElevenLabs API key](https://elevenlabs.io/) (optional — for
     higher-quality voice; Erebus falls back to the browser's built-in speech
     synthesis if you skip this).
3. **Configure your keys.** Copy `.env.example` to `.env` and fill in your keys,
   or just paste them into the Config panel in the browser once Erebus is
   running (that works too, but you'll need to re-enter them each session).
4. **Run it:**
   ```
   python server.py
   ```
   This starts the local server and opens Erebus in Chrome. On Windows, you can
   also just double-click `START_EREBUS.bat`.
5. **(Optional) Local speech recognition.** If you want fully offline, local
   speech-to-text instead of the browser's built-in recognizer:
   ```
   pip install faster-whisper sounddevice numpy
   python whisper_mic.py --test
   ```
   Speak and confirm it transcribes what you say, then run it normally
   (`python whisper_mic.py`) alongside `server.py`, and switch the interface's
   "Mic source" dropdown to Whisper.

## Notes

- Everything here runs locally except the two API calls (Anthropic for the
  model, ElevenLabs for voice) — your conversations, memory, and face data stay
  on your machine.
- Face recognition stores a 128-number mathematical descriptor per enrolled
  face, not a photo.
- This is a personal project, built and tuned around one person's setup and
  preferences (including a default English/Greek/Bulgarian/Russian character).
  Treat it as a reference architecture to adapt, not a polished product.
