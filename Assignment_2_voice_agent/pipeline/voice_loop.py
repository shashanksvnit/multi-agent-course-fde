"""
voice_loop.py  -  the turn loop (Layer A).

    mic -> VAD endpointing -> STT -> Agent -> TTS -> speakers

with per-stage latency timing so the room can SEE where the ~800ms turn budget
goes. Provider (Groq/OpenAI) is chosen in .env; see providers.py.

Modes:
    python voice_loop.py          # real mic
    python voice_loop.py --text   # type your turn (no audio deps / no mic)  -  always works
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time

from agent import Agent
from providers import make_provider

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass  # .env is optional; env vars still work. Keeps the offline mock zero-install.

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
ENDPOINT_SILENCE_MS = int(os.getenv("ENDPOINT_SILENCE_MS", "600"))


# --- Latency instrumentation: the point of the "measure the loop" segment ---

class Stopwatch:
    def __init__(self):
        self.marks: dict[str, float] = {}
        self._t = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.marks[stage] = (now - self._t) * 1000.0
        self._t = now

    def report(self) -> None:
        total = sum(self.marks.values())
        print("  ── turn latency ──")
        for stage, ms in self.marks.items():
            print(f"    {stage:<12} {ms:6.0f} ms")
        print(f"    {'TOTAL':<12} {total:6.0f} ms  (target < ~800 ms)")


# --- Audio (imported lazily so --text mode needs no audio libs) ---

def record_utterance() -> bytes:
    """Capture mic until the caller pauses (VAD endpointing). Returns 16-bit PCM."""
    import sounddevice as sd
    import webrtcvad

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_ms = 30
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)     # samples per frame
    silence_frames_needed = ENDPOINT_SILENCE_MS // frame_ms

    frames: list[bytes] = []
    started = False
    trailing_silence = 0

    print("  (listening  -  speak, then pause)")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=frame_len,
                           dtype="int16", channels=1) as stream:
        while True:
            block, _ = stream.read(frame_len)
            frame = bytes(block)
            if len(frame) < frame_len * 2:             # short tail frame
                continue
            speech = vad.is_speech(frame, SAMPLE_RATE)
            if speech:
                started = True
                trailing_silence = 0
                frames.append(frame)
            elif started:
                trailing_silence += 1
                frames.append(frame)
                if trailing_silence >= silence_frames_needed:
                    break
    return b"".join(frames)


def play_wav_bytes(wav: bytes) -> None:
    """Play WAV bytes via the configured local audio player."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(wav)
        f.flush()
        subprocess.run([os.getenv("AUDIO_PLAYER_CMD", "afplay"), f.name], check=False)


def speak(provider: Provider, text: str) -> None:
    """Speak `text`: cloud TTS returns audio, or the provider handles playback."""
    print(f"agent> {text}")
    audio = provider.synthesize(text)
    if audio:
        play_wav_bytes(audio)


# --- The loop ---

def run(text_mode: bool) -> None:
    provider = make_provider()
    agent = Agent(provider)
    print(f"Provider: {provider.name} | LLM: {provider.llm_model}")
    print("Call started. Say/type 'goodbye' or Ctrl-C to hang up.\n")

    speak(provider, "Thanks for calling Aurora Hotel reservations. How can I help?")

    while True:
        try:
            sw = Stopwatch()

            if text_mode:
                user_text = input("you> ")
            else:
                pcm = record_utterance()
                user_text = provider.transcribe(pcm, SAMPLE_RATE)
                print(f"you> {user_text}")
            sw.lap("stt")
            if not user_text.strip():
                continue

            reply, action = agent.respond(user_text)
            sw.lap("llm+tools")

            speak(provider, reply)
            sw.lap("tts")

            sw.report()
            print()

            if action == "hangup":
                print("[call ended  -  SIP BYE]")
                break
            if action == "transfer":
                print("[transferring to front desk  -  SIP REFER to front-desk]")
                break

        except (EOFError, KeyboardInterrupt):
            print("\n[caller hung up  -  SIP BYE]")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Workshop voice loop")
    parser.add_argument("--text", action="store_true",
                        help="type turns instead of speaking (no mic / no audio deps)")
    args = parser.parse_args()
    run(text_mode=args.text)


if __name__ == "__main__":
    main()
