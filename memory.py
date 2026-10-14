"""Memory stream: the agent's complete record of experience, plus the paper's
retrieval scoring = recency + importance + relevance (all in [0,1], weights=1).
"""

from dataclasses import dataclass, field

from llm import cosine


@dataclass
class Memory:
    id: int
    description: str
    kind: str  # seed | observation | reflection | plan | conversation
    created: float  # sim minutes
    last_accessed: float
    importance: int  # 1-10
    embedding: list
    evidence: list = field(default_factory=list)


class MemoryStream:
    def __init__(self, llm, decay=0.995):
        self.llm = llm
        self.memories = []
        self._next_id = 0
        self.decay = decay  # per sim-hour since last access

    def add(self, description, kind, now, evidence=None, importance=None):
        imp = (
            importance
            if importance is not None
            else self.llm.rate_importance(description)
        )
        emb = self.llm.embed([description])[0]
        m = Memory(self._next_id, description, kind, now, now, imp, emb, evidence or [])
        self._next_id += 1
        self.memories.append(m)
        return m

    def retrieve(self, query, now, n=8):
        if not self.memories:
            return []
        qe = self.llm.embed([query])[0]
        scored = []
        for m in self.memories:
            recency = self.decay ** ((now - m.last_accessed) / 60.0)
            relevance = cosine(qe, m.embedding)
            importance = m.importance / 10.0
            scored.append((recency + relevance + importance, m))
        top = [m for _, m in sorted(scored, key=lambda x: -x[0])[:n]]
        for m in top:
            m.last_accessed = now
        return top

    def recent(self, n=100):
        return self.memories[-n:]

    def by_kind(self, kind):
        return [m for m in self.memories if m.kind == kind]
