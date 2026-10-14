# Tiny Smallville

A lightweight reimplementation of **Generative Agents: Interactive Simulacra of
Human Behavior** (Park et al., UIST '23) — the paper where 25 LLM-powered Sims-style
characters plan their days, gossip about a mayoral candidate, and autonomously
organize a Valentine's Day party.

This version keeps the paper's full agent architecture but shrinks everything
else: **5 agents**, **6 locations**, **~1,200 lines of Python + JS**, and a
canvas UI you can watch in your browser. It runs on free LLM tiers
(Gemini, Ollama, OpenRouter) or fully offline with a deterministic mock.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # put your key in LLM_API_KEY (see "LLM providers")
.venv/bin/uvicorn server:app --port 8787
```

Open http://localhost:8787 — press **▶ Run** and watch the agents go about
their day. Click an agent to inspect its plan, memory stream, and reflections.

**No API key yet?** Run `MOCK_LLM=1 .venv/bin/uvicorn server:app --port 8787`.
The sim runs end-to-end on canned deterministic responses — useful for
verifying the plumbing before you spend tokens.

## What's implemented from the paper

| Paper component | This repo | Where |
|---|---|---|
| Memory stream: every observation stored as `{description, created, last_accessed, importance, embedding}` | ✅ | `memory.py` |
| Retrieval score = `recency_decay + importance + cosine(relevance)` | ✅ | `MemoryStream.retrieve` |
| Importance = LLM rates "poignancy" 1–10 per memory | ✅ | `LLM.rate_importance` |
| Reflection when recent importance sum > threshold; salient questions → cited insights | ✅ | `Agent.reflect` |
| Daily plan in broad strokes → recursively decomposed into ~10-min steps | ✅ (decomposed lazily per block) | `Agent.plan_day`, `_steps_for` |
| Reaction check on surprising observations ("the stove is burning") | ✅ | `Agent._maybe_react` |
| Natural-language dialogue conditioned on each agent's memories of the other | ✅ 4-turn exchanges | `Agent.converse` |
| Emergent info diffusion (party invite, mayoral candidacy) | ✅ seeded | `simulation.py` personas |
| Environment as a containment tree; object states agents can perceive | ✅ flat version | `world.py` |
| Phaser sprites, 25 agents, pathfinding, object state side-effects | ❌ out of scope | — |

## Architecture

```
        ┌──────────────────────────── each sim tick (10 game-min) ───────────────────────────┐
        │                                                                                    │
        ▼                                                                                    │
  perceive()  ── new object/agent states in current location ──► memory.add(observation)      │
        │                                                              │                     │
        ▼                                                    importance sum > 100?           │
  _maybe_react() ── "burning"/"party"/"leak" etc. ──► LLM: should I react? ──► pending_reaction
        │                                                                                    │
        ▼                                                                                    │
  current plan block ──► target location ──► move ──► arrive? action = 10-min step            │
        │                                                                                    │
        ▼                                                                                    │
  another agent in same location, not on cooldown? ──► converse() (4 turns)                   │
        │                                                                                    │
        └──────────────── both transcript + observations go back into memory stream ──────────┘
```

- **Time** — one tick = 10 simulated minutes (`SIM_MINUTES_PER_TICK`). Ticks run
  in a background thread; LLM calls are synchronous, so real-model pace is
  bounded by API latency and rate limits.
- **Movement** — each plan block picks a wander-point inside the target
  location's rect; the agent slides toward it. No pathfinding — it's a toy.
- **Dialogue memory** — both participants store the transcript summary as a
  `conversation` memory, so a party invite heard by Maria retrievable later by
  Klaus is exactly how info propagates.
- **User intervention** — the UI lets you set any object's state
  (`lin_house:stove → burning`) or speak to an agent as its *inner voice*
  (stored as an importance-8 observation, per the paper's trick for directives).

## Files

```
llm.py         provider-agnostic LLM client (chat, embed, importance) + MOCK_LLM
memory.py      MemoryStream + retrieval scoring
agent.py       perceive / plan / react / reflect / converse — the paper's core
world.py       6 locations, object states, grid geometry
simulation.py  personas, clock, event log, interventions
server.py      FastAPI: /state /step /run /pause /intervene /agent/<name>
static/index.html  canvas map + agent inspector + intervene panel
```

## LLM providers

All providers go through an OpenAI-compatible endpoint — swapping is a
`.env` change:

```env
LLM_PROVIDER=gemini      # gemini | openai | ollama | openrouter
LLM_API_KEY=...
# LLM_CHAT_MODEL=...     # optional override
# LLM_EMBED_MODEL=...    # optional; no embeddings -> keyword-vector fallback
```

| Provider | Free? | Notes |
|---|---|---|
| `gemini` | ✅ generous free tier | default; get a key at aistudio.google.com, no billing needed |
| `openai` | paid | gpt-4o-mini default |
| `ollama` | ✅ local | `brew install ollama && ollama pull qwen3:8b` — slower but unlimited |
| `openrouter` | some free models | falls back to keyword relevance (no embeddings) |

Every memory write costs one importance-rating call, so the sim is chatty:
roughly a few hundred calls per simulated day with 5 agents. Gemini's free
~15 RPM tier handles it; raise `TICK_SECONDS` if you hit 429s.

## Seeded scenario

As in the paper, two pieces of information start in exactly one head each:

- **Isabella** owns Hobbs Cafe and *"is planning a Valentine's Day party at
  5pm today and wants to invite everyone she meets"*.
- **Sam** *"is running for mayor and wants to tell everyone about his
  candidacy"*.

Nothing else is scripted. Whether the party propagates through dialogue — and
whether anyone shows up at 17:00 — depends on the model, the retrieval scores,
and luck, which is rather the point.

## Known simplifications

- Dialogue is a fixed 4 turns rather than agents choosing to end it.
- Reactions don't mutate object states (Isabella won't actually extinguish
  the stove — she'll just *say* she did).
- No persistence: restarting the server resets the world.
- Agents can't enter buildings or learn new locations; the map is known upfront.

## References

- [Generative Agents (UIST '23 paper)](https://doi.org/10.1145/3586183.3606763)
- [Original Stanford implementation](https://github.com/joonspk-research/generative_agents)
  (full 25-agent version with Phaser + Django — heavier but more faithful)
