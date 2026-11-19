"""narva-assay core: rerun a Narva store item's test on your own conversations, under Narva's protocol.

The protocol is the one behind Narva's round 1 (bench DOM-01-phaseB-v2.1): same renderer, same answer prompt,
same judge prompt, same baseline (each turn one memory with its session date, top 10 turns by vector
similarity), same paired bootstrap (10,000 resamples, seed 0). Only the data, the models and your system change.
"""
import asyncio
import hashlib
import importlib
import json
import os
import random
import re
import tempfile
import time

from .refmem import RefMem

# ---------------- prompts (identical to Narva's runner; a test checks this) ----------------
ANSWER_PROMPT = """You answer questions about a long conversation between two people, using only the memory
given below. Dates in session headers tell you when things were said; resolve relative dates
("last week") against them. Answer in a short phrase. If the memory does not contain the answer,
say "not mentioned".

MEMORY:
{context}

QUESTION: {question}
ANSWER:"""

JUDGE_PROMPT = """You grade an answer to a question about a conversation.
Question: {question}
Gold answer: {gold}
Generated answer: {answer}
The generated answer is CORRECT if it conveys the same fact as the gold answer, even if it is
shorter, longer or worded differently; for dates, the same day, month or period counts as correct.
Otherwise it is WRONG. Reply with exactly one word: CORRECT or WRONG."""


# ---------------- data ----------------
def load_data(path, limit=None):
    """LoCoMo JSON (a list of samples with 'conversation' and 'qa') or Narva JSONL (one conversation per line:
    {"id", "sessions": [{"date", "turns": [{"speaker", "text"}]}], "questions": [{"question", "answer", "category"?, "time"?}]}).
    Returns conversations as (id, turns, questions); LoCoMo category 5 (no gold answer) is left out, as in Narva's runs."""
    raw = open(path, "rb").read()
    convs = []
    if path.endswith(".jsonl"):
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            turns, n = [], 0
            for s_i, s in enumerate(c["sessions"], 1):
                for t in s["turns"]:
                    n += 1
                    turns.append({"id": f"S{s_i}:{n}", "session": s_i, "date": s.get("date", ""), "speaker": t["speaker"], "text": t["text"]})
            qs = [dict(q, time=bool(q.get("time"))) for q in c["questions"]]
            convs.append((c["id"], turns, qs))
    else:
        for i, sample in enumerate(json.loads(raw)):
            conv, turns, s = sample["conversation"], [], 1
            while f"session_{s}" in conv:
                date = conv.get(f"session_{s}_date_time", "")
                for t in conv[f"session_{s}"]:
                    text = t.get("text", "")
                    if t.get("blip_caption"):
                        text += f" [shares an image: {t['blip_caption']}]"
                    turns.append({"id": t.get("dia_id"), "session": s, "date": date, "speaker": t["speaker"], "text": text})
                s += 1
            qs = [{"question": q["question"], "answer": str(q["answer"]), "category": q.get("category"), "time": q.get("category") == 2}
                  for q in sample["qa"] if q.get("category") != 5 and "answer" in q]
            convs.append((sample.get("sample_id", f"conv{i}"), turns, qs))
    return convs[:limit] if limit else convs, hashlib.sha256(raw).hexdigest()


def render(turns):
    lines, last = [], None
    for t in turns:
        if t["session"] != last:
            lines.append(f"\n--- Session {t['session']} ({t['date']}) ---")
            last = t["session"]
        lines.append(f"[{t['id']}] {t['speaker']}: {t['text']}")
    return "\n".join(lines)


# ---------------- models ----------------
class Gemini:
    """Gemini through google-genai: Vertex AI when GOOGLE_CLOUD_PROJECT is set, else the Gemini API with GEMINI_API_KEY."""
    def __init__(self, model, embed_model="text-embedding-005", flex=False, timeout_s=900):
        from google import genai
        from google.genai import types
        self.t, self.model, self.embed_model = types, model, embed_model
        headers = {"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": "flex"} if flex else None
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            ho = types.HttpOptions(headers=headers) if headers else None
            self.c = genai.Client(vertexai=True, project=project, location="global", http_options=ho)
            self.e = genai.Client(vertexai=True, project=project, location=os.environ.get("EMBED_LOCATION", "us-central1"))
        else:
            self.c = self.e = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.timeout_ms = timeout_s * 1000

    def generate(self, prompt, max_tokens):
        t = self.t
        cfg = t.GenerateContentConfig(temperature=0, max_output_tokens=max_tokens, http_options=t.HttpOptions(timeout=self.timeout_ms),
                                      thinking_config=t.ThinkingConfig(thinking_level="LOW"))
        r = self.c.models.generate_content(model=self.model, contents=prompt, config=cfg)
        u = r.usage_metadata
        return (r.text or "").strip(), (u.prompt_token_count or 0), (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 100):
            out += [e.values for e in self.e.models.embed_content(model=self.embed_model, contents=texts[i:i + 100]).embeddings]
        return out


class OpenAICompatible:
    """Any OpenAI-compatible endpoint (OPENAI_BASE_URL, OPENAI_API_KEY): chat completions and embeddings."""
    def __init__(self, model, embed_model="text-embedding-3-small", timeout_s=900):
        self.model, self.embed_model, self.timeout = model, embed_model, timeout_s
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ["OPENAI_API_KEY"]

    def _post(self, path, body):
        import urllib.request
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())

    def generate(self, prompt, max_tokens):
        r = self._post("/chat/completions", {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                                             "messages": [{"role": "user", "content": prompt}]})
        u = r.get("usage", {})
        return (r["choices"][0]["message"]["content"] or "").strip(), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 100):
            out += [d["embedding"] for d in self._post("/embeddings", {"model": self.embed_model, "input": texts[i:i + 100]})["data"]]
        return out


class Fake:
    """No network, no cost: checks the plumbing. Answers with the context's first quoted line; judges by word overlap."""
    def __init__(self, model="fake", **_):
        self.model = model

    def generate(self, prompt, max_tokens):
        if prompt.startswith("You grade"):
            gold = re.search(r"Gold answer: (.*)", prompt).group(1).lower().split()
            ans = re.search(r"Generated answer: (.*)", prompt).group(1).lower()
            return ("CORRECT" if gold and sum(w in ans for w in gold) * 2 >= len(gold) else "WRONG"), len(prompt) // 4, 1
        q = re.search(r"QUESTION: (.*)", prompt).group(1).lower().split()
        mem = [ln for ln in prompt.split("\n") if ln.startswith("[")]
        best = max(mem, key=lambda ln: sum(w in ln.lower() for w in q), default="not mentioned")
        return best.split(": ", 1)[-1][:200], len(prompt) // 4, 20

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v)
        return out


def make_model(kind, model, **kw):
    return {"gemini": Gemini, "openai": OpenAICompatible, "fake": Fake}[kind](model, **kw)


# ---------------- systems: prepare(conversation) then context(question) ----------------
class MemorySystem:
    """Plug your own memory in by subclassing this and passing --system your_module:YourClass.
    prepare() gets every turn of one conversation in order, each with its session date; context() returns the
    text the answer model will see for one question. Keep each conversation separate (cid)."""
    name = "system"

    def __init__(self, model):
        self.model = model

    def prepare(self, cid, turns):
        raise NotImplementedError

    def context(self, cid, question):
        raise NotImplementedError


class Baseline(MemorySystem):
    """Narva's vector-only baseline: each turn one memory with its session date; top 10 turns, shown in conversation order."""
    name, hybrid = "baseline", False

    def prepare(self, cid, turns):
        self.mem, self.turns = getattr(self, "mem", {}), getattr(self, "turns", {})
        texts = [f"{t['date']} {t['speaker']}: {t['text']}" for t in turns]
        m = RefMem(os.path.join(tempfile.mkdtemp(), "m.db"), self.model.embed, supersede="append", hybrid=self.hybrid)
        for t, v in zip(texts, self.model.embed(texts)):
            m.write("a", t, vector=v)
        self.mem[cid], self.turns[cid] = m, turns

    def context(self, cid, question):
        got = self.mem[cid].recall("a", question, k=10)
        return render([self.turns[cid][g["id"] - 1] for g in sorted(got, key=lambda g: g["id"])])


class Hybrid(Baseline):
    """The baseline with keyword search fused in (BM25 + vector, reciprocal rank fusion), as in Narva's I3 test."""
    name, hybrid = "hybrid", True


class FullContext(MemorySystem):
    """No memory system: the whole conversation, every session with its date, is the context for every question."""
    name = "fullcontext"

    def prepare(self, cid, turns):
        self.ctx = getattr(self, "ctx", {})
        self.ctx[cid] = render(turns)

    def context(self, cid, question):
        return self.ctx[cid]


BUILTIN = {"baseline": Baseline, "hybrid": Hybrid, "fullcontext": FullContext}


def make_system(spec, model):
    if spec in BUILTIN:
        return BUILTIN[spec](model)
    mod, _, cls = spec.partition(":")
    s = getattr(importlib.import_module(mod), cls)(model)
    s.name = getattr(s, "name", None) or cls
    return s


# ---------------- run: answers saved as they come, a rerun asks only what is missing ----------------
async def _call(fn, *a, tries=8):
    for i in range(tries):
        try:
            return await asyncio.to_thread(fn, *a)
        except Exception as e:
            if i == tries - 1:
                return f"ERROR {repr(e)[:150]}", 0, 0
            await asyncio.sleep(min(120, 2 ** i + 2) * (0.5 + random.random()))


def _load(path):
    return {(r["conversation"], r["q_index"]): r for r in json.load(open(path))} if os.path.exists(path) else {}


def _save(path, rows):
    tmp = path + ".tmp"
    json.dump(list(rows), open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


async def answer_and_judge(system, convs, answer_model, judge_model, out_dir, parallel=8, label=None):
    label = label or system.name
    path = os.path.join(out_dir, f"answers_{label}.json")
    got = {k: r for k, r in _load(path).items() if not r["answer"].startswith("ERROR") and r.get("verdict") in ("CORRECT", "WRONG")}
    jobs = [(cid, i, q) for cid, _, qs in convs for i, q in enumerate(qs) if (cid, i) not in got]
    need = {cid for cid, _, _ in jobs}
    for cid, turns, _ in convs:
        if cid in need:
            await asyncio.to_thread(system.prepare, cid, turns)
    sem, lock = asyncio.Semaphore(parallel), asyncio.Lock()

    async def one(cid, i, q):
        async with sem:
            ctx = await asyncio.to_thread(system.context, cid, q["question"])
            ans, tin, tout = await _call(answer_model.generate, ANSWER_PROMPT.format(context=ctx, question=q["question"]), 2048)
            if ans.startswith("ERROR"):
                return
            jp = JUDGE_PROMPT.format(question=q["question"], gold=q["answer"], answer=ans)
            v, jin, jout = await _call(judge_model.generate, jp, 512)
            if v.strip().upper() not in ("CORRECT", "WRONG"):   # an empty verdict (cap spent on thinking) is asked again, larger cap
                v, jin2, jout2 = await _call(judge_model.generate, jp, 4096)
                jin, jout = jin + jin2, jout + jout2
            v = v.strip().upper()
            if v not in ("CORRECT", "WRONG"):
                return
        async with lock:
            got[(cid, i)] = {"system": label, "conversation": cid, "q_index": i, "question": q["question"], "gold": q["answer"],
                             "category": q.get("category"), "time": q.get("time", False), "answer": ans, "verdict": v,
                             "correct": int(v == "CORRECT"), "prompt_tokens": tin, "output_tokens": tout,
                             "judge_tokens": jin + jout, "context_chars": len(ctx)}
            if len(got) % 50 == 0:
                _save(path, sorted(got.values(), key=lambda r: (r["conversation"], r["q_index"])))
    for _ in range(3):
        todo = [j for j in jobs if (j[0], j[1]) not in got]
        if not todo:
            break
        await asyncio.gather(*(one(*j) for j in todo))
    _save(path, sorted(got.values(), key=lambda r: (r["conversation"], r["q_index"])))
    missing = sum(1 for j in jobs if (j[0], j[1]) not in got)
    if missing:
        raise SystemExit(f"{label}: {missing} answers or verdicts still failed after 3 passes; good ones are saved, rerun to finish")
    return got


# ---------------- statistics (identical to Narva's) ----------------
def paired_ci(d, n=10000, seed=0):
    rnd = random.Random(seed)
    m = len(d)
    means = sorted(sum(d[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {"mean": round(sum(d) / m, 4), "lo": round(means[int(0.025 * n)], 4), "hi": round(means[int(0.975 * n) - 1], 4), "n_items": m}


def compare(a, b, keep=lambda r: True):
    keys = sorted(k for k in a if k in b and keep(a[k]))
    if not keys:
        raise SystemExit("no paired questions to compare")
    ci = paired_ci([a[k]["correct"] - b[k]["correct"] for k in keys])
    ci["verdict"] = "Works" if ci["lo"] > 0 else ("Harms" if ci["hi"] < 0 else "No effect")
    return ci


def accuracy(rows):
    return round(sum(r["correct"] for r in rows.values()) / len(rows), 4)


def foreign_years(rows, turns_by_conv):
    """Time answers naming a year that appears in no session date of their conversation (Narva's ingest-year check, generalised)."""
    hits = 0
    for r in rows.values():
        if not r.get("time"):
            continue
        years = set(re.findall(r"\b(19\d\d|20\d\d)\b", " ".join(t["date"] for t in turns_by_conv[r["conversation"]])))
        if years and set(re.findall(r"\b(19\d\d|20\d\d)\b", r["answer"])) - years:
            hits += 1
    return hits, sum(1 for r in rows.values() if r.get("time"))


def agreement(a, b):
    keys = [k for k in a if k in b]
    return round(sum(a[k]["verdict"] == b[k]["verdict"] for k in keys) / len(keys), 4), len(keys)


def estimate(convs, systems, answer_model_name):
    """Prompt tokens before any call, at about 4 characters a token (an estimate, printed as one)."""
    head = len(ANSWER_PROMPT)
    out = {}
    for s in systems:
        tot = 0
        for _, turns, qs in convs:
            per = len(render(turns)) if s == "fullcontext" else (1600 if s in ("baseline", "hybrid") else 2000)
            tot += len(qs) * (head + per) // 4
        out[s] = {"answer_prompt_tokens_est": tot, "questions": sum(len(q) for _, _, q in convs)}
    return out


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
