"""
lib/llm.py — the LLM translation call
======================================
One job: turn an English string into Mexican Spanish using an LLM.

Provider: OpenAI (`pip install openai`, set OPENAI_API_KEY).

  - The PROMPT pins the register to Mexican Spanish (es-MX), not
    generic/Castilian Spanish. It asks for ONLY the translation, no preamble.
  - Numbers, prices ($), and product/model codes are kept unchanged.
  - The returned string is cleaned (quotes/whitespace the model may add are stripped).

FAIL LOUD: the call is not wrapped in a try/except that returns `text` on error.
If the provider fails, the exception propagates so the caller returns a 502.
Silently returning the untranslated input is an automatic fail on this
assignment (and a real production bug — it ships English while looking healthy).
"""
import os

from openai import AsyncOpenAI

MODEL_DEFAULT = os.getenv("MODEL", "gpt-4o-mini")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Lazily construct the client so it reads OPENAI_API_KEY after .env is loaded."""
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's English text into "
    "natural, conversational MEXICAN Spanish (es-MX) — the register used in Mexico, "
    "not generic/Castilian Spanish (e.g. use \"tú\"/\"ustedes\" conventions and Mexican "
    "vocabulary, never \"vosotros\"). Return ONLY the translation: no preamble, no notes, "
    "no wrapping quotes. Keep numbers, prices (with their currency symbol), and product "
    "or model codes exactly as written in the source."
)


async def translate_text(text: str, target: str = "es-MX", model: str = MODEL_DEFAULT) -> str:
    """Return `text` translated into `target` (Mexican Spanish by default).

    Errors are not caught here: a provider failure must propagate so the
    caller can surface a 502 instead of silently serving English.
    """
    resp = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip().strip('"').strip("'")
