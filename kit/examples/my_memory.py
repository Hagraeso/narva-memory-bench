"""A template for plugging your own memory into narva-assay. Run with:
   narva-assay run DOM-01-S16 --data locomo10.json --system my_memory:KeepLastTurns --backend gemini ... --yes
(run from this folder, or put the module on PYTHONPATH)."""
from narva_assay.core import MemorySystem, render


class KeepLastTurns(MemorySystem):
    """A deliberately weak example: remembers only the last 30 turns of each conversation."""
    name = "keep_last_turns"

    def prepare(self, cid, turns):
        self.store = getattr(self, "store", {})
        self.store[cid] = turns[-30:]          # turns arrive in order, each with its session date

    def context(self, cid, question):
        return render(self.store[cid])         # the text the answer model sees
