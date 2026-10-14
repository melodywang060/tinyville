"""FastAPI server: serves the sim state + the canvas UI, runs the tick loop
in a background thread when the user presses Run."""

import os
import threading
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from simulation import Simulation

TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1.0"))

app = FastAPI()
sim = Simulation()

_lock = threading.Lock()
_running = threading.Event()


def _loop():
    while _running.is_set():
        with _lock:
            sim.tick(1)
        time.sleep(TICK_SECONDS)


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/state")
def state():
    with _lock:
        return sim.state()


@app.post("/step")
def step(n: int = 1):
    with _lock:
        sim.tick(min(n, 20))
    return {"ok": True}


@app.post("/run")
def run():
    if not _running.is_set():
        _running.set()
        threading.Thread(target=_loop, daemon=True).start()
    return {"ok": True}


@app.post("/pause")
def pause():
    _running.clear()
    return {"ok": True}


@app.get("/agent/{name}")
def agent(name: str):
    with _lock:
        info = sim.agent_info(name)
    return info or JSONResponse({"error": "no such agent"}, status_code=404)


class Intervention(BaseModel):
    kind: str  # "object" | "tell"
    target: str = ""
    value: str = ""
    agent: str = ""
    text: str = ""


@app.post("/intervene")
def intervene(body: Intervention):
    with _lock:
        ok = sim.intervene(body.kind, target=body.target, value=body.value,
                           agent=body.agent, text=body.text)
    return {"ok": ok}
