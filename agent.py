"""Generative agent: perceive -> retrieve -> plan/react -> reflect.

Mirrors the paper's architecture at toy scale:
- memory stream with scored retrieval (memory.py)
- daily plan generated in broad strokes, lazily decomposed into 10-min steps
- reaction check on surprising observations
- periodic reflection when recent importance exceeds a threshold
- natural-language dialogue between co-located agents
"""

import os
import re

from memory import MemoryStream

TICK_MIN = float(os.getenv("SIM_MINUTES_PER_TICK", "10"))
REFLECT_THRESHOLD = float(os.getenv("REFLECT_THRESHOLD", "100"))
TALK_COOLDOWN_MIN = 90
MOVE_PER_TICK = 1.5

EMOJI_RULES = [
    ("sleep", "💤"), ("bed", "💤"), ("breakfast", "🍳"), ("dinner", "🍽"),
    ("lunch", "🥪"), ("eat", "🍽"), ("coffee", "☕"), ("cook", "🍳"),
    ("stove", "🔥"), ("talk", "💬"), ("convers", "💬"), ("chat", "💬"),
    ("walk", "🚶"), ("read", "📖"), ("study", "📚"), ("write", "✍️"),
    ("research", "📝"), ("work", "💼"), ("shop", "🛒"), ("party", "🎉"),
    ("decorat", "🎈"), ("clean", "🧹"), ("shower", "🚿"), ("medic", "💊"),
    ("pharmacy", "💊"), ("paint", "🎨"), ("exercise", "🏃"), ("music", "🎵"),
]


def fmt(minutes):
    m = int(minutes) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def emoji_for(action):
    a = action.lower()
    for k, e in EMOJI_RULES:
        if k in a:
            return e
    return "🙂"


class Agent:
    def __init__(self, name, seed, home, color, llm, world, now):
        self.name = name
        self.seed = seed
        self.home = home
        self.color = color
        self.llm = llm
        self.world = world
        self.mem = MemoryStream(llm)
        for s in seed.split(";"):
            s = s.strip()
            if s:
                self.mem.add(s, "seed", now, importance=6)

        self.x, self.y = world.anchor(home)
        self.tx, self.ty = self.x, self.y
        self.location = home
        self.plan = []          # [{"start","end","loc","desc","steps"}]
        self.action = "is waking up"
        self.pending_reaction = None
        self.busy_until = 0.0
        self.reflect_acc = 0.0
        self.last_seen = {}     # "obj key"/"person:name" -> last description
        self.last_talk = {}     # other name -> sim minutes
        self.last_action_mem = None
        self.planned_day = None  # day number plan was generated for

    # ---------------- prompts / summaries ---------------------------------

    def summary(self):
        recent_reflections = self.mem.by_kind("reflection")[-3:]
        extra = " ".join(m.description for m in recent_reflections)
        return f"Name: {self.name}\n{self.seed}\n{extra}".strip()

    def _mem_summary(self, query, now, n=6):
        mems = self.mem.retrieve(query, now, n)
        return "\n".join(f"- {m.description}" for m in mems) or "(nothing relevant)"

    # ---------------- planning --------------------------------------------

    def plan_day(self, now, date_str):
        day = int(now // (24 * 60))
        if self.planned_day == day:
            return
        prompt = (
            f"{self.summary()}\n\nToday is {date_str}. "
            f"Locations: {self.world.keys_prompt()}. "
            f"Use 'home' for {self.home}.\n"
            f"Here is {self.name}'s plan today in broad strokes, as lines in "
            f"the format 'HH:MM | location | activity' starting at wake-up and "
            f"ending at bedtime:\n"
        )
        out = self.llm.chat(prompt, max_tokens=400)
        plan = self._parse_plan(out, now)
        self.plan = plan
        self.planned_day = day
        desc = "; ".join(f"{fmt(b['start'])} {b['desc']}" for b in plan)
        self.mem.add(f"{self.name}'s plan for today: {desc}", "plan", now,
                   importance=6)

    def _parse_plan(self, text, now):
        blocks = []
        for line in text.splitlines():
            m = re.match(r"\s*(?:[-*•]|\d+[.)])?\s*(\d{1,2}):(\d{2})\s*[|\-–]\s*([^|\-–]+)\s*[|\-–]\s*(.+)", line)
            if not m:
                continue
            start = int(m.group(1)) * 60 + int(m.group(2))
            loc = self.world.resolve(m.group(3)) or self.home
            blocks.append({"start": start, "loc": loc,
                           "desc": m.group(4).strip(), "steps": None})
        if not blocks:
            blocks = [
                {"start": 8 * 60, "loc": self.home, "desc": "wake up and get ready", "steps": None},
                {"start": 10 * 60, "loc": "johnson_park", "desc": "take a walk", "steps": None},
                {"start": 12 * 60, "loc": "hobbs_cafe", "desc": "have lunch", "steps": None},
                {"start": 15 * 60, "loc": "hobbs_cafe", "desc": "spend the afternoon", "steps": None},
                {"start": 19 * 60, "loc": self.home, "desc": "have dinner and relax", "steps": None},
                {"start": 23 * 60, "loc": self.home, "desc": "go to sleep", "steps": None},
            ]
        blocks.sort(key=lambda b: b["start"])
        for i, b in enumerate(blocks):
            b["end"] = blocks[i + 1]["start"] if i + 1 < len(blocks) else 24 * 60
        return blocks

    def _block_for(self, now):
        if not self.plan:
            return None
        t = now % (24 * 60)
        cur = self.plan[0]
        for b in self.plan:
            if b["start"] <= t:
                cur = b
        return cur

    def _steps_for(self, block):
        if block["steps"] is not None:
            return block["steps"]
        prompt = (
            f"{self.summary()}\n\nBetween {fmt(block['start'])} and "
            f"{fmt(block['end'])}, {self.name} plans to: {block['desc']} "
            f"at {block['loc']}.\nDecompose this into steps of about 10 "
            f"minutes. Format: 'HH:MM | action', one per line.\n"
        )
        out = self.llm.chat(prompt, max_tokens=300)
        steps = []
        for line in out.splitlines():
            m = re.match(r"\s*(?:[-*•]|\d+[.)])?\s*(\d{1,2}):(\d{2})\s*[|\-–]\s*(.+)", line)
            if m:
                steps.append({"start": int(m.group(1)) * 60 + int(m.group(2)),
                              "desc": m.group(3).strip()})
        if not steps:
            steps = [{"start": block["start"] + i * 30, "desc": block["desc"]}
                     for i in range(max(1, int((block["end"] - block["start"]) // 30)))]
        block["steps"] = sorted(steps, key=lambda s: s["start"])
        return block["steps"]

    def _step_for(self, block, now):
        steps = self._steps_for(block)
        t = now % (24 * 60)
        cur = steps[0]
        for s in steps:
            if s["start"] <= t:
                cur = s
        return cur["desc"]

    # ---------------- perception ------------------------------------------

    def perceive(self, agents, now):
        """Collect changed object/agent states in the current location."""
        loc = self.world.location_at(self.x, self.y)
        if loc:
            self.location = loc
        new_obs = []
        for key, state in self.world.objects_in(self.location):
            loc_key, obj = key.split(":", 1)
            label = self.world.locations[loc_key]["label"]
            desc = f"The {obj} at {label} is {state}"
            if self.last_seen.get(key) != desc:
                self.last_seen[key] = desc
                new_obs.append(desc)
        for other in agents:
            if other is self:
                continue
            if other.location == self.location:
                desc = f"{other.name} {other.action}"
                k = f"person:{other.name}"
                if self.last_seen.get(k) != desc:
                    self.last_seen[k] = desc
                    new_obs.append(desc)
        for desc in new_obs:
            m = self.mem.add(desc, "observation", now)
            self.reflect_acc += m.importance
        return new_obs

    # ---------------- reaction --------------------------------------------

    def _maybe_react(self, obs, now):
        interesting = [
            o for o in obs
            if any(w in o.lower() for w in
                   ("burning", "leak", "empty", "party", "invit", "shout", "argu"))
        ]
        if not interesting:
            return
        prompt = (
            f"{self.summary()}\n\nIt is {fmt(now)}.\n"
            f"{self.name}'s status: {self.name} {self.action}.\n"
            f"Observation: {interesting[0]}\n"
            f"Summary of relevant context from {self.name}'s memory:\n"
            f"{self._mem_summary(interesting[0], now)}\n\n"
            f"Should {self.name} react to the observation? Answer 'no' to "
            f"continue, or 'yes: <what {self.name} does next>'."
        )
        out = self.llm.chat(prompt, max_tokens=120)
        if out.lower().startswith("yes"):
            reaction = out.split(":", 1)[1].strip() if ":" in out else out
            self.pending_reaction = reaction
            self.mem.add(f"{self.name} {reaction}", "observation", now)
            return reaction
        return None

    # ---------------- reflection ------------------------------------------

    def reflect(self, now):
        recent = self.mem.recent(100)
        stmts = "\n".join(f"{i+1}. {m.description}" for i, m in enumerate(recent))
        q_out = self.llm.chat(
            f"{stmts}\n\nGiven only the information above, what are 3 most "
            f"salient high-level questions we can answer about the subjects "
            f"in the statements? Format: '1) ...', one per line."
        )
        questions = re.findall(r"\d+\)\s*(.+)", q_out)[:3] or [
            f"What is {self.name} focused on right now?"
        ]
        for q in questions:
            mems = self.mem.retrieve(q, now, n=6)
            evidence = "\n".join(f"{i+1}. {m.description}" for i, m in enumerate(mems))
            out = self.llm.chat(
                f"Statements about {self.name}\n{evidence}\n\nWhat 3 high-level "
                f"insights can you infer from the above statements? "
                f"(example format: insight (because of 1, 5, 3))",
                max_tokens=250,
            )
            for line in out.splitlines():
                line = line.strip().lstrip("-").strip()
                if not line:
                    continue
                ids = [int(n) - 1 for n in re.findall(r"\d+", line.split("because of")[-1])]
                ev = [mems[i].id for i in ids if 0 <= i < len(mems)]
                insight = line.split("(because")[0].strip()
                self.mem.add(insight, "reflection", now, evidence=ev)

    # ---------------- dialogue --------------------------------------------

    def converse(self, other, now, date_str):
        transcript = []
        speakers = [self, other]
        for turn in range(4):
            spk = speakers[turn % 2]
            if self.llm.mock:
                from llm import MOCK_LINES
                line = MOCK_LINES[turn % len(MOCK_LINES)]
            else:
                history = "\n".join(
                    f"{n}: {l}" for n, l in transcript
                ) or "(conversation just started)"
                prompt = (
                    f"{spk.summary()}\n\nIt is {date_str}, {fmt(now)}.\n"
                    f"{spk.name}'s status: {spk.name} {spk.action}.\n"
                    f"{spk.name} is talking with {speakers[(turn+1)%2].name}.\n"
                    f"Summary of relevant context from {spk.name}'s memory:\n"
                    f"{spk._mem_summary(f'{speakers[(turn+1)%2].name} and current events', now)}\n\n"
                    f"Here is the dialogue history:\n{history}\n\n"
                    f"What would {spk.name} say next? Reply with just the "
                    f"spoken line, or END to end the conversation."
                )
                line = self.llm.chat(prompt, max_tokens=80)
                if not line or line.strip().upper() == "END":
                    break
                line = re.sub(rf"^{re.escape(spk.name)}\s*:\s*", "", line).strip('" ')
            transcript.append((spk.name, line))
        if transcript:
            text = "; ".join(f"{n} said '{l}'" for n, l in transcript)
            summary_line = f"{self.name} and {other.name} had a conversation: {text}"
            for a in speakers:
                a.mem.add(summary_line, "conversation", now)
                a.busy_until = now + TICK_MIN * 2
                a.last_talk[[o.name for o in speakers if o is not a][0]] = now
                a.last_seen[f"person:{other.name}"] = f"{other.name} {other.action}"
        return transcript

    # ---------------- tick ------------------------------------------------

    def act(self, agents, now, date_str, events):
        obs = self.perceive(agents, now)
        if self.reflect_acc >= REFLECT_THRESHOLD:
            self.reflect_acc = 0
            self.reflect(now)
            events.append(f"💭 {self.name} reflected")
        self._maybe_react(obs, now)

        if now < self.busy_until:
            return

        block = self._block_for(now)
        if block:
            if block is not getattr(self, "_cur_block", None):
                self._cur_block = block
                self.tx, self.ty = self.world.anchor(block["loc"])
            dx, dy = self.tx - self.x, self.ty - self.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > MOVE_PER_TICK:
                self.x += dx / dist * MOVE_PER_TICK
                self.y += dy / dist * MOVE_PER_TICK
                self.action = f"is walking to {self.world.locations[block['loc']]['label']}"
            else:
                self.x, self.y = self.tx, self.ty
                if self.pending_reaction:
                    self.action = self.pending_reaction
                    self.pending_reaction = None
                else:
                    self.action = f"is {self._step_for(block, now)}"
                loc = self.world.location_at(self.x, self.y)
                if loc:
                    self.location = loc
        elif self.pending_reaction:
            self.action = self.pending_reaction
            self.pending_reaction = None

        if self.action != self.last_action_mem:
            self.mem.add(f"{self.name} {self.action}", "observation", now,
                       importance=3)
            self.last_action_mem = self.action

        # strike up a conversation with someone sharing the location
        for other in agents:
            if other is self or other.location != self.location:
                continue
            if self.name > other.name:  # let the lower-named agent initiate
                continue
            if now < other.busy_until:
                continue
            if now - self.last_talk.get(other.name, -999) < TALK_COOLDOWN_MIN:
                continue
            if now - other.last_talk.get(self.name, -999) < TALK_COOLDOWN_MIN:
                continue
            transcript = self.converse(other, now, date_str)
            if transcript:
                events.append(f"💬 {self.name} & {other.name}: " +
                              " / ".join(l for _, l in transcript[:2]))
            break

    def info(self, now):
        t = int(now % (24 * 60))
        return {
            "name": self.name,
            "summary": self.seed,
            "location": self.location,
            "action": self.action,
            "plan": [
                {"time": f"{fmt(b['start'])}-{fmt(b['end'])}",
                 "loc": b["loc"], "desc": b["desc"],
                 "active": b["start"] <= t < b["end"]}
                for b in self.plan
            ],
            "memories": [
                {"time": fmt(m.created), "kind": m.kind,
                 "importance": m.importance, "text": m.description}
                for m in self.mem.memories[-40:]
            ],
            "reflections": [m.description for m in self.mem.by_kind("reflection")],
        }
