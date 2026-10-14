"""Provider-agnostic LLM client.

All providers are reached through an OpenAI-compatible endpoint, so one code
path serves Gemini, OpenAI, Ollama, and OpenRouter. Set MOCK_LLM=1 to run the
whole sim deterministically with zero API calls.
"""

import hashlib
import math
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "chat": "gemini-2.5-flash-lite",
        "embed": "text-embedding-004",
    },
    "openai": {
        "base_url": None,
        "chat": "gpt-4o-mini",
        "embed": "text-embedding-3-small",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "chat": "qwen3:8b",
        "embed": "nomic-embed-text",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "chat": "meta-llama/llama-3.3-70b-instruct",
        "embed": "",  # no embeddings -> keyword fallback
    },
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def keyword_vec(text, dim=256):
    """Bag-of-words hash vector. Used as a relevance fallback when the provider
    has no embeddings API."""
    v = [0.0] * dim
    for tok in re.findall(r"[a-z']+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    return v


class LLM:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini")
        cfg = PROVIDERS.get(self.provider, PROVIDERS["gemini"])
        self.chat_model = os.getenv("LLM_CHAT_MODEL", cfg["chat"])
        self.embed_model = os.getenv("LLM_EMBED_MODEL", cfg["embed"])
        self.mock = os.getenv("MOCK_LLM") == "1"
        self.calls = 0
        self._embed_ok = bool(self.embed_model)
        if self.mock:
            return
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY") or "no-key",
            base_url=os.getenv("LLM_BASE_URL") or cfg["base_url"] or None,
        )

    def chat(self, prompt, system=None, max_tokens=500, temperature=0.7, retries=4):
        self.calls += 1
        if self.mock:
            return mock_chat(prompt)
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        for i in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(min(30, 3 * 2**i))

    def embed(self, texts):
        if self.mock or not self._embed_ok:
            return [keyword_vec(t) for t in texts]
        try:
            r = self.client.embeddings.create(model=self.embed_model, input=texts)
            return [d.embedding for d in r.data]
        except Exception:
            self._embed_ok = False
            return [keyword_vec(t) for t in texts]

    def rate_importance(self, text):
        """The paper's poignancy rating: ask the model for a 1-10 integer."""
        out = self.chat(
            "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing "
            "teeth, making bed) and 10 is extremely poignant (e.g., a break up, "
            "college acceptance), rate the likely poignancy of the following "
            f"piece of memory.\nMemory: {text}\nRating: ",
            max_tokens=8,
            temperature=0,
        )
        m = re.search(r"\d+", out or "")
        n = int(m.group()) if m else 4
        return max(1, min(10, n))


# ---------------------------------------------------------------------------
# Mock mode: deterministic canned responses keyed on prompt content. Enough to
# exercise every code path (planning, reaction, dialogue, reflection) offline.


def _h(prompt):
    return int(hashlib.md5(prompt.encode()).hexdigest(), 16)


MOCK_PLAN = """8:00 | home | wake up and do the morning routine
9:00 | hobbs_cafe | work and chat with people who stop by
12:30 | johnson_park | take a walk and eat lunch
13:30 | hobbs_cafe | keep working through the afternoon
17:00 | hobbs_cafe | hang out at the cafe
18:30 | home | make dinner and relax
23:00 | home | go to sleep"""


MOCK_LINES = [
    "Hey! Great to see you. How's your day going?",
    "Pretty good! I'm planning a party at the cafe at 5pm — you should come.",
    "Oh nice, a party? I'd love to, I'll try to make it.",
    "Wonderful, see you there! It's going to be so fun.",
    "Talk to you later then!",
]


def mock_chat(prompt):
    p = prompt.lower()
    h = _h(prompt)

    if "poignancy" in p:
        return str(2 + h % 8)
    if "salient high-level questions" in p or "high-level questions" in p:
        return (
            "1) What are they working on?\n"
            "2) Who do they spend time with?\n"
            "3) What do they care about?"
        )
    if "high-level insights" in p:
        return (
            "- They are friendly and enjoy meeting neighbors (because of 1, 2)\n"
            "- They care about community events (because of 2, 3)"
        )
    if "plan today" in p or "broad strokes" in p:
        return MOCK_PLAN
    if "which area should" in p:
        for key in ("hobbs_cafe", "johnson_park", "willows_market",
                    "oak_hill_dorm", "lin_house", "moore_house"):
            if key in prompt:
                return key
        return "home"
    if "dialogue history" in p:
        n = prompt.count("\n", prompt.find("dialogue history"))
        return MOCK_LINES[n % len(MOCK_LINES)]
    if "should" in p and "react" in p:
        return "no"
    if "decompose" in p or "into steps" in p:
        return ""
    return "is going about their day"
