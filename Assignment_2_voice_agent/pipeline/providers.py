"""
providers.py  -  one adaptor, two backends: Groq and OpenAI.

Groq speaks the OpenAI API dialect, so a single code path covers both  -  only
base_url, api_key, and model names differ. Switch with PROVIDER=groq|openai in
.env; move to your OpenAI key later by flipping that one value.

Exposes three stages the voice loop needs:
    chat(messages, tools)        -> LLM turn (OpenAI-style tool calling)
    transcribe(pcm_int16, rate)  -> STT (Whisper)
    synthesize(text)             -> TTS; returns WAV bytes, or None if it
                                    already played via the system voice command
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import wave
from types import SimpleNamespace as NS

# Sensible defaults per backend. Any of these can be overridden in .env.
PRESETS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # 70b = reliable tool-calling; swap to llama-3.1-8b-instant for lower latency.
        "llm_model": "llama-3.3-70b-versatile",
        "stt_model": "whisper-large-v3-turbo",
        "tts_model": "canopylabs/orpheus-v1-english",
        "tts_voice": "troy",
    },
    "openai": {
        "base_url": None,                # SDK default endpoint
        "api_key_env": "OPENAI_API_KEY",
        "llm_model": "gpt-4o-mini",
        "stt_model": "whisper-1",
        "tts_model": "tts-1",
        "tts_voice": "alloy",
    },
}


class Provider:
    """Configured client for one backend. Read from .env on construction."""

    def __init__(self, name: str | None = None):
        name = (name or os.getenv("PROVIDER", "groq")).lower()
        if name not in PRESETS:
            raise ValueError(f"Unknown PROVIDER {name!r}; use one of {list(PRESETS)}")
        self.name = name
        p = PRESETS[name]

        api_key = os.getenv(p["api_key_env"])
        if not api_key:
            raise RuntimeError(f"Set {p['api_key_env']} in your .env (PROVIDER={name})")
        from openai import OpenAI  # lazy: the mock path needs no SDK installed
        self.client = OpenAI(api_key=api_key, base_url=p["base_url"])

        # Per-stage overrides fall back to the preset.
        self.llm_model = os.getenv("LLM_MODEL") or p["llm_model"]
        self.stt_model = os.getenv("STT_MODEL") or p["stt_model"]
        self.tts_model = os.getenv("TTS_MODEL") or p["tts_model"]
        self.tts_voice = os.getenv("TTS_VOICE") or p["tts_voice"]
        self.tts_instructions = os.getenv("TTS_INSTRUCTIONS")
        # "provider" = cloud TTS; "system" = local system voice command.
        self.tts_backend = os.getenv("TTS_BACKEND", "provider").lower()

    # --- LLM ---
    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """One chat-completion call. Returns the raw SDK response."""
        return self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            temperature=0.3,
        )

    # --- STT ---
    def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw 16-bit mono PCM via Whisper."""
        wav = _pcm_to_wav(pcm_int16, sample_rate)
        wav.name = "turn.wav"  # SDK infers format from the filename
        resp = self.client.audio.transcriptions.create(
            model=self.stt_model,
            file=wav,
            response_format="text",
        )
        return (resp if isinstance(resp, str) else resp.text).strip()

    # --- TTS ---
    def synthesize(self, text: str) -> bytes | None:
        """Return WAV bytes for `text`, or None if played directly by the OS."""
        if self.tts_backend == "system":
            subprocess.run([os.getenv("SYSTEM_TTS_CMD", "say"), text], check=False)
            return None
        speech_args = {
            "model": self.tts_model,
            "voice": self.tts_voice,
            "input": text,
            "response_format": "wav",
        }
        if self.tts_instructions:
            speech_args["instructions"] = self.tts_instructions
        resp = self.client.audio.speech.create(
            **speech_args,
        )
        return resp.content


# --- audio helpers ---

def _pcm_to_wav(pcm_int16: bytes, sample_rate: int) -> io.BytesIO:
    """Wrap raw 16-bit mono PCM samples into an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm_int16)
    buf.seek(0)
    return buf


# --- Mock backend: full offline end-to-end, no network / key / SDK ---

class MockProvider:
    """Drop-in stand-in for Provider. Rule-based LLM, scripted STT, no-op TTS.

    Same interface (chat / transcribe / synthesize) so voice_loop.py and
    agent.py can't tell the difference. Use for rehearsals, CI, and testing the
    loop without touching Groq/OpenAI. Enable with PROVIDER=mock.
    """

    name = "mock"

    def __init__(self):
        self.llm_model = "mock-llm"
        self.stt_model = "mock-stt"
        self.tts_model = "mock-tts"
        self.tts_voice = "mock"
        self.tts_backend = os.getenv("TTS_BACKEND", "print").lower()
        # Scripted transcripts for mic mode (there's no offline STT); cycles.
        self._stt_script = [
            "I need a room from August 12 to August 14 for two guests.",
            "Book it for Priya Shah, priya@example.com.",
            "Can I speak to a person?",
            "Goodbye",
        ]
        self._stt_i = 0

    def chat(self, messages: list[dict], tools=None):
        """Rule-based reply mimicking OpenAI-style tool calling."""
        last = messages[-1]
        # After a tool ran, speak a reply built from its result.
        if last.get("role") == "tool":
            result = last["content"]
            if result.lower().startswith("available rooms"):
                return _mk_text(f"{result} Would you like me to book one of these?")
            if result.lower().startswith("booking confirmed"):
                return _mk_text(result)
            return _mk_text(result)  # transfer / hangup / not-found: speak as-is

        text = (last.get("content") or "").lower()
        if any(w in text for w in ("human", "person", "representative", "agent", "operator")):
            return _mk_tool("transfer_to_human", {})
        if any(w in text for w in ("bye", "goodbye", "that's all", "thats all",
                                   "nothing else", "no thanks", "hang up")):
            return _mk_tool("end_call", {})
        if any(w in text for w in ("weather", "news", "sports", "stock", "joke", "trivia")):
            return _mk_text("I can only help with hotel reservations. Are you looking to book, change, or cancel a stay?")
        if any(w in text for w in ("change", "cancel", "modify", "front desk")):
            return _mk_tool("transfer_to_human", {})
        if any(w in text for w in ("book", "reserve", "yes", "confirm")) and any(
            w in text for w in ("name", "email", "@", "phone", "priya", "shah")
        ):
            return _mk_tool("create_booking", {
                "check_in": "August 12",
                "check_out": "August 14",
                "guests": 2,
                "room_type": "standard",
                "guest_name": "Priya Shah",
                "contact": "priya@example.com",
            })
        if any(w in text for w in ("room", "hotel", "stay", "book", "reservation", "guests", "guest")):
            return _mk_tool("check_availability", {
                "check_in": "August 12",
                "check_out": "August 14",
                "guests": 2,
                "room_type": "standard",
            })
        return _mk_text("I can help with hotel reservations only. Would you like to book, change, or cancel a stay?")

    def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        """No offline STT  -  return the next scripted phrase (rehearsal mode)."""
        phrase = self._stt_script[self._stt_i % len(self._stt_script)]
        self._stt_i += 1
        return phrase

    def synthesize(self, text: str) -> bytes | None:
        """No cloud TTS. Optionally use a local voice command; else print-only."""
        if self.tts_backend == "system":
            subprocess.run([os.getenv("SYSTEM_TTS_CMD", "say"), text], check=False)
        return None  # voice_loop already prints the agent's text


def _mk_text(content: str):
    return NS(choices=[NS(message=NS(content=content, tool_calls=None))])


def _mk_tool(name: str, args: dict):
    tc = NS(id=f"call_{name}", type="function",
            function=NS(name=name, arguments=json.dumps(args)))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))])


def make_provider(name: str | None = None):
    """Factory: returns MockProvider for PROVIDER=mock, else a live Provider."""
    name = (name or os.getenv("PROVIDER", "groq")).lower()
    if name == "mock":
        return MockProvider()
    return Provider(name)
