"""
EREBUS - Local Speech-to-Text (Whisper)
=======================================
Replaces Chrome's Web Speech API (which sends your audio to Google, and does not
exist in Chromium on a Raspberry Pi) with transcription that runs on THIS machine.

Same file runs on Windows and on the Pi — only MODEL_SIZE changes.

RUN IT ALONE FIRST to check your mic and the model:
    python whisper_mic.py --test
Speak. It prints what it hears. Ctrl+C to stop.

Then run it normally alongside server.py:
    python whisper_mic.py

INSTALL:
    pip install faster-whisper sounddevice numpy

First run downloads the model (needs internet, once). After that it is offline.
"""

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ── CONFIG ────────────────────────────────────────────────────────────────
# tiny  = fastest, weak on Greek/Bulgarian
# base  = middle ground
# small = good on Greek/Bulgarian
# large-v3 = the best Whisper makes; excellent Greek/Bulgarian.
#            Fast on an NVIDIA GPU, slow on CPU-only.   <-- default (you have an RTX 4050)
MODEL_SIZE = "large-v3"

# Whisper's auto-detect sometimes guesses a language that isn't even being spoken —
# Portuguese and Chinese are its classic "unsure" misfires on short/noisy audio.
# We keep auto-detect (so you can switch freely between these four just by speaking),
# but throw away any result whose detected language is outside this set.
# ISO codes: en=English, el=Greek, bg=Bulgarian, ru=Russian.
ALLOWED_LANGUAGES = {"en", "el", "bg", "ru"}
# If the detected language IS in the set but Whisper is very unsure AND the clip is
# tiny, it's usually noise — drop it. (Longer clips are trusted even at lower conf.)
MIN_LANG_CONFIDENCE = 0.50


PORT = 8766           # erebus.html connects here (server.py stays on 8765)
SAMPLE_RATE = 16000   # what Whisper expects
BLOCK = 480           # 30ms per audio block

# Speech detection. Raise SILENCE_MULT if it triggers on room noise;
# lower it if quiet speech gets missed.
SILENCE_MULT = 3.0        # speech = this many times louder than the calibrated room
ABS_FLOOR = 0.005         # ignore anything below this no matter how quiet the room
START_BLOCKS = 3          # ~90ms of sound before we call it speech
END_SILENCE = 0.8         # seconds of quiet that ends an utterance
MIN_UTTERANCE = 0.35      # shorter than this = a cough, discard
MAX_UTTERANCE = 20.0      # hard stop so one noise can't record forever
PREROLL = 0.3             # seconds kept from BEFORE the trigger, so the first
                          # syllable isn't clipped

# Whisper invents text from silence — usually subtitle boilerplate. Drop these.
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.",
    "you", "bye.", ".", "so", "uh", "um", "okay.",
    "субтитры сделал dimatorzok", "продолжение следует...",
    "ευχαριστώ.", "благодаря.",
}

# ── STATE ─────────────────────────────────────────────────────────────────
audio_q = queue.Queue()
clients = []           # SSE subscribers (the browser)
clients_lock = threading.Lock()
paused = threading.Event()   # set = do not listen (Erebus is talking)
running = True
model = None
noise_floor = 0.01


def log(msg):
    print(f"[WHISPER] {msg}", flush=True)


def broadcast(payload):
    """Push a transcription to every connected browser."""
    line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
    with clients_lock:
        dead = []
        for c in clients:
            try:
                c.wfile.write(line)
                c.wfile.flush()
            except Exception:
                dead.append(c)
        for d in dead:
            clients.remove(d)


def rms(block):
    """Loudness of one audio block."""
    import numpy as np
    return float(np.sqrt(np.mean(np.square(block))))


def is_junk(text):
    t = text.strip().lower()
    if not t:
        return True
    if t in HALLUCINATIONS:
        return True
    # a "sentence" of only punctuation/symbols
    if not any(ch.isalnum() for ch in t):
        return True
    return False


# ── AUDIO CAPTURE + SEGMENTATION ──────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if status:
        pass  # overflows are common and harmless
    audio_q.put(indata[:, 0].copy())


def calibrate(sd_stream, seconds=1.0):
    """Measure the room so the threshold suits YOUR mic, not a guess."""
    global noise_floor
    import numpy as np
    levels = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            block = audio_q.get(timeout=0.5)
            levels.append(rms(block))
        except queue.Empty:
            break
    if levels:
        noise_floor = max(float(np.median(levels)), 0.0005)
    log(f"Room noise: {noise_floor:.5f} -> speech above {max(noise_floor * SILENCE_MULT, ABS_FLOOR):.5f}")


def listen_loop(on_text):
    """Capture -> detect an utterance -> transcribe -> hand off the text."""
    import numpy as np

    threshold = max(noise_floor * SILENCE_MULT, ABS_FLOOR)
    preroll_blocks = int((PREROLL * SAMPLE_RATE) / BLOCK)
    preroll = deque(maxlen=preroll_blocks)

    speaking = False
    loud_run = 0
    quiet_run = 0
    voiced_blocks = 0        # blocks of ACTUAL sound, not silence
    buf = []
    end_blocks = int((END_SILENCE * SAMPLE_RATE) / BLOCK)
    max_blocks = int((MAX_UTTERANCE * SAMPLE_RATE) / BLOCK)
    keep_tail = int((0.2 * SAMPLE_RATE) / BLOCK)   # a little trailing silence helps Whisper

    while running:
        try:
            block = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # While Erebus is talking we throw audio away, so he never hears himself.
        if paused.is_set():
            if speaking:
                speaking, buf, loud_run, quiet_run, voiced_blocks = False, [], 0, 0, 0
            preroll.clear()
            continue

        level = rms(block)
        loud = level > threshold

        if not speaking:
            preroll.append(block)
            if loud:
                loud_run += 1
                if loud_run >= START_BLOCKS:
                    speaking = True
                    buf = list(preroll)      # include what came just before
                    preroll.clear()
                    quiet_run = 0
                    voiced_blocks = loud_run
            else:
                loud_run = 0
        else:
            buf.append(block)
            if loud:
                quiet_run = 0
                voiced_blocks += 1
            else:
                quiet_run += 1

            if quiet_run >= end_blocks or len(buf) >= max_blocks:
                speaking = False
                loud_run = 0
                # Drop most of the trailing silence — it costs transcription time
                # and gives Whisper more room to hallucinate.
                if quiet_run > keep_tail:
                    buf = buf[:-(quiet_run - keep_tail)]
                audio = np.concatenate(buf) if buf else np.array([], dtype="float32")
                buf = []
                # Judge length on REAL speech, not the padding. Without this a
                # 0.15s cough plus preroll and tail looks like a 1.1s sentence.
                voiced_seconds = voiced_blocks * BLOCK / SAMPLE_RATE
                dur = len(audio) / SAMPLE_RATE
                voiced_blocks = 0
                if voiced_seconds < MIN_UTTERANCE:
                    continue
                threading.Thread(target=transcribe_and_emit,
                                 args=(audio, dur, on_text), daemon=True).start()


def transcribe_and_emit(audio, dur, on_text):
    t0 = time.time()
    try:
        segments, info = model.transcribe(
            audio,
            beam_size=1,                      # fast; beam 5 is slower for little gain
            language=None,                    # auto-detect (EN/GR/BG/RU)
            condition_on_previous_text=False, # stops repeat-loop hallucinations
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text for s in segments).strip()
    except Exception as e:
        log(f"transcribe failed: {e}")
        return

    took = time.time() - t0
    if is_junk(text):
        log(f"(ignored {dur:.1f}s of noise)")
        return
    lang = getattr(info, "language", "?")
    conf = getattr(info, "language_probability", 0) or 0

    # Option A: keep auto-detect, but reject anything outside our four languages.
    if lang not in ALLOWED_LANGUAGES:
        log(f'(ignored {lang} {conf:.0%}: "{text[:40]}") — outside allowed languages')
        return
    # A quiet blip detected as one of our languages but with low confidence is
    # usually noise slipping through. Trust longer clips even at lower confidence.
    if conf < MIN_LANG_CONFIDENCE and dur < 1.2:
        log(f'(ignored low-confidence {lang} {conf:.0%}: "{text[:40]}")')
        return

    log(f'[{lang} {conf:.0%}] "{text}"  ({dur:.1f}s audio, {took:.1f}s to transcribe)')
    on_text({"text": text, "language": lang, "confidence": round(conf, 3),
             "duration": round(dur, 2), "took": round(took, 2)})


# ── HTTP / SSE ────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ready": model is not None,
                "model": MODEL_SIZE,
                "paused": paused.is_set(),
                "noise_floor": round(noise_floor, 5),
            }).encode())
            return

        if self.path == "/stream":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            with clients_lock:
                clients.append(self)
            log("browser connected")
            try:
                while running:
                    time.sleep(15)
                    self.wfile.write(b": keepalive\n\n")   # stops proxies timing out
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                with clients_lock:
                    if self in clients:
                        clients.remove(self)
                log("browser disconnected")
            return

        self.send_response(404)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path in ("/pause", "/resume"):
            if self.path == "/pause":
                paused.set()
            else:
                paused.clear()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"paused": paused.is_set()}).encode())
            return
        self.send_response(404)
        self._cors()
        self.end_headers()


class ThreadedHTTP(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ── MAIN ──────────────────────────────────────────────────────────────────
def add_nvidia_dll_paths():
    """
    On Windows, the pip-installed NVIDIA libraries (cublas, cudnn) put their DLLs
    inside site-packages\\nvidia\\*\\bin, but Windows doesn't look there at runtime,
    so faster-whisper fails with 'cublas64_12.dll is not found'. This registers
    those folders with the DLL loader so the GPU path actually works.
    Safe no-op on Linux / when the folders don't exist.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import os
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        nvidia_root = list(spec.submodule_search_locations)[0]
        added = []
        for sub in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, sub, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)     # Python 3.8+ on Windows
                    added.append(sub)
                except Exception:
                    pass
                # also prepend to PATH as a belt-and-suspenders fallback
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if added:
            print(f"[WHISPER] Registered GPU libraries: {', '.join(added)}", flush=True)
    except Exception as e:
        print(f"[WHISPER] Could not register NVIDIA DLLs ({e}) — GPU may fall back to CPU", flush=True)


def main():
    global model, running

    add_nvidia_dll_paths()   # must run BEFORE faster-whisper loads the CUDA backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="just print what you say (no server) — use this first")
    ap.add_argument("--model", default=MODEL_SIZE, help="tiny | base | small | large-v3")
    ap.add_argument("--device-type", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto (GPU if present), or force cuda / cpu")
    ap.add_argument("--list-mics", action="store_true")
    ap.add_argument("--device", type=int, default=None, help="mic index from --list-mics")
    args = ap.parse_args()

    try:
        import numpy  # noqa
        import sounddevice as sd
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run:  pip install faster-whisper sounddevice numpy")
        sys.exit(1)

    if args.list_mics:
        print(sd.query_devices())
        return

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Missing faster-whisper.  Run:  pip install faster-whisper")
        sys.exit(1)

    print("=" * 52)
    print("  EREBUS - Local Speech-to-Text (Whisper)")
    print("=" * 52)

    # Start the HTTP server FIRST. Loading 'small' downloads ~460MB on the first
    # run, and if the port isn't open during that, Erebus just reports "cannot
    # reach the service" with no idea anything is happening.
    srv = None
    if not args.test:
        try:
            srv = ThreadedHTTP(("", PORT), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            log(f"Listening for Erebus on http://localhost:{PORT}")
        except OSError as e:
            log(f"Could not open port {PORT}: {e}")
            log("Another copy of whisper_mic.py is probably already running.")
            sys.exit(1)

    log(f"Loading '{args.model}' model...")
    log("(first run downloads it — large-v3 is ~3GB, needs internet, once)")
    t0 = time.time()

    # Use the NVIDIA GPU if it's there (transcription in a fraction of a second);
    # fall back to CPU cleanly if CUDA isn't available or fails to init.
    def load_on(device, compute):
        return WhisperModel(args.model, device=device, compute_type=compute)

    want = args.device_type
    model_obj = None
    if want in ("auto", "cuda"):
        try:
            model_obj = load_on("cuda", "float16")   # float16 = fast + fits 6GB VRAM
            log("Running on GPU (CUDA, float16)")
        except Exception as e:
            if want == "cuda":
                log(f"GPU was requested but failed: {e}")
            else:
                log(f"No usable GPU ({str(e)[:60]}...) — falling back to CPU")
    if model_obj is None:
        try:
            model_obj = load_on("cpu", "int8")
            log("Running on CPU (int8)")
        except Exception as e:
            log(f"Could not load the model: {e}")
            log("Check your internet connection for the first download.")
            sys.exit(1)
    model = model_obj
    log(f"Model ready in {time.time() - t0:.1f}s")

    if args.test:
        def on_text(p):
            print(f'\n  >>> "{p["text"]}"   [{p["language"]}]\n')
    else:
        on_text = broadcast

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, channels=1,
                            dtype="float32", callback=audio_callback,
                            device=args.device):
            log("Calibrating room noise — stay quiet for a second...")
            calibrate(None)
            log("Listening. Speak normally.  (Ctrl+C to stop)")
            listen_loop(on_text)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log(f"Audio error: {e}")
        log("Try:  python whisper_mic.py --list-mics    then --device N")
    finally:
        running = False
        log("Stopped.")


if __name__ == "__main__":
    main()
