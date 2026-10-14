"""The world: a flat grid with labeled rectangular areas containing objects
with mutable string states. Objects are addressed as "location_key:object".
"""

import random

GRID_W, GRID_H = 24, 23

LOCATIONS = {
    "hobbs_cafe": {
        "label": "Hobbs Cafe",
        "rect": [1, 1, 9, 6],
        "objects": {
            "coffee machine": "idle",
            "espresso counter": "clean",
            "pastry case": "full of pastries",
        },
    },
    "willows_market": {
        "label": "Willows Market & Pharmacy",
        "rect": [12, 1, 10, 6],
        "objects": {
            "pharmacy counter": "staffed",
            "shelves": "stocked",
            "register": "idle",
        },
    },
    "lin_house": {
        "label": "The Lin House",
        "rect": [1, 9, 9, 6],
        "objects": {
            "stove": "off",
            "bed": "made",
            "shower": "off",
            "desk": "tidy",
        },
    },
    "johnson_park": {
        "label": "Johnson Park",
        "rect": [12, 9, 10, 6],
        "objects": {
            "gazebo": "empty",
            "pond": "calm",
            "benches": "empty",
        },
    },
    "oak_hill_dorm": {
        "label": "Oak Hill Dorm",
        "rect": [1, 16, 9, 5],
        "objects": {
            "common room": "quiet",
            "study desk": "free",
            "kitchenette": "clean",
        },
    },
    "moore_house": {
        "label": "The Moore House",
        "rect": [12, 16, 10, 5],
        "objects": {
            "stove": "off",
            "bed": "made",
            "armchair": "empty",
        },
    },
}


class World:
    def __init__(self):
        self.locations = LOCATIONS
        self.states = {
            f"{loc}:{obj}": state
            for loc, d in LOCATIONS.items()
            for obj, state in d["objects"].items()
        }
        self._anchors = {k: self._center(d["rect"]) for k, d in LOCATIONS.items()}
        self._rng = random.Random(7)

    def _center(self, r):
        return [r[0] + r[2] / 2, r[1] + r[3] / 2]

    def anchor(self, loc):
        """A wander-point inside the location's rect (stable-ish per agent)."""
        r = self.locations[loc]["rect"]
        x = self._rng.uniform(r[0] + 1, r[0] + r[2] - 1)
        y = self._rng.uniform(r[1] + 1, r[1] + r[3] - 1)
        return [x, y]

    def location_at(self, x, y):
        for k, d in self.locations.items():
            rx, ry, rw, rh = d["rect"]
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return k
        return None

    def state_of(self, key):
        return self.states.get(key)

    def set_state(self, key, value):
        if key in self.states:
            self.states[key] = value
            return True
        return False

    def objects_in(self, loc):
        return [
            (f"{loc}:{obj}", self.states[f"{loc}:{obj}"])
            for obj in self.locations[loc]["objects"]
        ]

    def describe_objects(self, loc):
        return [
            f"the {obj} is {self.states[f'{loc}:{obj}']}"
            for obj in self.locations[loc]["objects"]
        ]

    def resolve(self, text):
        """Fuzzy-map free text to a location key."""
        t = text.strip().lower().replace(" ", "_")
        if t in self.locations:
            return t
        for k, d in self.locations.items():
            if t and (t in k or k in t or t in d["label"].lower()):
                return k
        return None

    def keys_prompt(self):
        return ", ".join(
            f"{k} ({d['label']})" for k, d in self.locations.items()
        )
