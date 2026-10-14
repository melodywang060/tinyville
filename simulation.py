"""The simulation: a clock, a world, and a handful of agents."""

import os

from agent import Agent, fmt, emoji_for
from llm import LLM
from world import World

TICK_MIN = float(os.getenv("SIM_MINUTES_PER_TICK", "10"))
START_MIN = 8 * 60  # simulation starts at 08:00 on day 0
START_DATE = "February 13, 2026"

PERSONAS = [
    {
        "name": "Isabella Rodriguez",
        "home": "hobbs_cafe",
        "color": "#e05555",
        "seed": (
            "Isabella Rodriguez owns Hobbs Cafe and loves making people feel "
            "welcome; Isabella is planning a Valentine's Day party at Hobbs "
            "Cafe at 5pm today and wants to invite everyone she meets; "
            "Isabella is warm, chatty, and loves decorating; Isabella's close "
            "friend is Maria Lopez"
        ),
    },
    {
        "name": "Maria Lopez",
        "home": "oak_hill_dorm",
        "color": "#e0a040",
        "seed": (
            "Maria Lopez is a student at Oak Hill College studying physics; "
            "Maria is Isabella Rodriguez's close friend and often hangs out "
            "at Hobbs Cafe; Maria has a secret crush on Klaus Mueller; "
            "Maria is shy but kind"
        ),
    },
    {
        "name": "Klaus Mueller",
        "home": "oak_hill_dorm",
        "color": "#5080d8",
        "seed": (
            "Klaus Mueller is a student at Oak Hill College studying "
            "sociology; Klaus is writing a research paper on gentrification; "
            "Klaus likes to read and take walks in Johnson Park; Klaus is "
            "friendly with Maria Lopez"
        ),
    },
    {
        "name": "John Lin",
        "home": "lin_house",
        "color": "#40a060",
        "seed": (
            "John Lin is a pharmacist at the Willows Market and Pharmacy; "
            "John loves helping customers; John lives at the Lin house; "
            "John likes discussing local politics"
        ),
    },
    {
        "name": "Sam Moore",
        "home": "moore_house",
        "color": "#9040c0",
        "seed": (
            "Sam Moore is a retired navy officer; Sam is running for mayor in "
            "the local election and wants to tell everyone about his "
            "candidacy; Sam lives at the Moore house; Sam likes chatting "
            "with neighbors"
        ),
    },
]


class Simulation:
    def __init__(self):
        self.llm = LLM()
        self.world = World()
        self.now = float(START_MIN)
        self.day = 0
        self.events = []
        self.agents = [
            Agent(p["name"], p["seed"], p["home"], p["color"],
                  self.llm, self.world, self.now)
            for p in PERSONAS
        ]
        for a in self.agents:
            a.plan_day(self.now, self.date_str())
        self.events.append(f"Simulation started at {self.date_str()}, {fmt(self.now)}")

    def date_str(self):
        return f"February {13 + self.day}, 2026"

    def tick(self, n=1):
        for _ in range(n):
            prev_day = self.day
            self.now += TICK_MIN
            self.day = int(self.now // (24 * 60))
            if self.day != prev_day:
                for a in self.agents:
                    a.plan = []
                    a.plan_day(self.now, self.date_str())
            for a in self.agents:
                a.act(self.agents, self.now, self.date_str(), self.events)
        self.events = self.events[-300:]

    def state(self):
        return {
            "time": fmt(self.now),
            "date": self.date_str(),
            "llm_calls": self.llm.calls,
            "mock": self.llm.mock,
            "locations": {
                k: {"label": d["label"], "rect": d["rect"]}
                for k, d in self.world.locations.items()
            },
            "objects": dict(self.world.states),
            "agents": [
                {
                    "name": a.name,
                    "x": a.x,
                    "y": a.y,
                    "location": a.location,
                    "action": a.action,
                    "emoji": emoji_for(a.action),
                    "color": a.color,
                }
                for a in self.agents
            ],
            "events": self.events[-40:],
        }

    def agent_info(self, name):
        for a in self.agents:
            if a.name == name:
                return a.info(self.now)
        return None

    def intervene(self, kind, **kw):
        if kind == "object":
            key = kw.get("target", "")
            value = kw.get("value", "")
            if self.world.set_state(key, value):
                self.events.append(f"🔧 {key} -> {value}")
                return True
            return False
        if kind == "tell":
            name, text = kw.get("agent", ""), kw.get("text", "")
            for a in self.agents:
                if a.name == name:
                    a.mem.add(f"Inner voice told {a.name}: {text}", "observation",
                              self.now, importance=8)
                    self.events.append(f"🗣 told {name}: {text}")
                    return True
            return False
        return False
