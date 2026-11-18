"""Narva phase B runner (spec DOM-01-phaseB-v2.0, round 1): LoCoMo recall accuracy for real memory
products under one protocol, against Narva's vector-only baseline and (phase D) the whole conversation
in the context window. Runs as narva-tester on the VM.

Every model call goes through clients wrapped here, so tokens, traffic type (Flex) and spend are
measured from the responses, including the calls Mem0 and Graphiti make internally. A cost guard
stops the run when the measured spend passes the spec's cap.
Env: SPEC_URI, EXPECTED_SPEC_SHA, SYSTEMS (comma list: baseline,fullcontext,mem0,graphiti), RUN_ID (reuse to add
systems to one run), FALKOR_HOST, N_CONVERSATIONS (default 10).
"""
import asyncio
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import tempfile
import threading
import time
import uuid

from google import genai
from google.cloud import storage
from google.genai import types

from refmem import RefMem

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "narva-rnd-engine")
BUCKET = os.environ.get("RUNS_BUCKET", "narva-rnd-engine-runs")
SYSTEMS = [s for s in os.environ.get("SYSTEMS", "baseline,mem0").split(",") if s]
SMOKE = os.environ.get("SMOKE") == "1"   # small real run: 1 conversation, 3 sessions, 20 questions, every code path
CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
N_CONV = int(os.environ.get("N_CONVERSATIONS", "10"))
ANSWER, LITE, PRO = "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"
RUN_ID = os.environ.get("RUN_ID") or ("smoke-" if SMOKE else "") + "phaseB-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
BASE = f"tests/{RUN_ID}"
gcs = storage.Client(project=PROJECT)
runs = gcs.bucket(BUCKET)
FLEX = {"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": "flex"}


def log(msg, **kw):
    print(json.dumps({"severity": "INFO", "message": msg, "run_id": RUN_ID, **kw}, default=str), flush=True)


def put(name, obj):
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=1)
    runs.blob(f"{BASE}/{name}").upload_from_string(data, content_type="application/json")


# ---------------- metered model clients ----------------
class Meter:
    def __init__(self, prices, cap_sek):
        self.p, self.cap = prices, cap_sek
        self.lock = threading.Lock()
        self.by = {}  # (bucket, model) -> [calls, in, out, non_flex, cached_in, sek]
        self.extra = {}  # bucket -> SEK not seen in responses (context cache storage)
        self.bucket = "setup"

    def add(self, model, um):
        tin = getattr(um, "prompt_token_count", 0) or 0
        tout = (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        flex = str(getattr(um, "traffic_type", "")).upper().endswith("ON_DEMAND_FLEX")
        cached = getattr(um, "cached_content_token_count", 0) or 0
        with self.lock:
            k = (self.bucket, model)
            v = self.by.setdefault(k, [0, 0, 0, 0, 0, 0.0])
            v[0] += 1; v[1] += tin; v[2] += tout; v[3] += (0 if flex else 1); v[4] += cached
            # priced per call: a call that was not served on Flex pays the full price; cached input tokens
            # pay the spec's cached-input factor of the full input price, never the Flex discount
            a, z = self.price(model)
            f = self.p["flex_factor"] if flex else 1.0
            v[5] += ((tin - cached) * a * f + cached * a * self.p["cached_input_factor"] + tout * z * f) / 1e6 * self.p["usd_to_sek"]
        if self.sek() > self.cap:
            raise SystemExit(f"cost guard: measured spend {self.sek():.0f} SEK passed the cap {self.cap} SEK")

    def price(self, model):
        p = self.p
        if "lite" in model:
            return p["lite_in"], p["lite_out"]
        if "pro" in model:
            return p["pro_in"], p["pro_out"]
        return p["flash_in"], p["flash_out"]

    def sek(self, bucket=None):
        tot = sum(v[5] for (b, m), v in list(self.by.items()) if not bucket or b == bucket)
        return tot + sum(x for b, x in list(self.extra.items()) if not bucket or b == bucket)

    def report(self):
        out = {f"{b}|{m}": {"calls": v[0], "in": v[1], "out": v[2], "non_flex": v[3], "cached_in": v[4], "sek": round(v[5], 2)}
               for (b, m), v in self.by.items()}
        out.update({f"{b}|extra": {"sek": round(x, 2)} for b, x in self.extra.items()})
        return out


METER = None
SPEC = {}


def metered(client):
    """Wrap a genai client so every generate_content (sync and async) is metered."""
    sync, asy = client.models.generate_content, client.aio.models.generate_content

    def g(*a, **k):
        r = sync(*a, **k)
        METER.add(k.get("model") or (a[0] if a else "?"), r.usage_metadata)
        return r

    async def ag(*a, **k):
        r = await asy(*a, **k)
        METER.add(k.get("model") or (a[0] if a else "?"), r.usage_metadata)
        return r
    client.models.generate_content, client.aio.models.generate_content = g, ag
    return client


def clients():
    llm = metered(genai.Client(vertexai=True, project=PROJECT, location="global",
                               http_options=types.HttpOptions(headers=FLEX)))
    emb = genai.Client(vertexai=True, project=PROJECT, location="us-central1")
    return llm, emb


LLM, EMB = None, None
SEM = asyncio.Semaphore(int(os.environ.get("CONCURRENCY", "64")))
RSEM = asyncio.Semaphore(int(os.environ.get("RETRIEVAL_CONCURRENCY", "8")))
# phase D sends ~20k-token prompts; at 64 in flight the service answered 429 for most calls (round 1 attempt 1)
FSEM = asyncio.Semaphore(int(os.environ.get("FULLCONTEXT_CONCURRENCY", "12")))
CALL_TIMEOUT_MS = 15 * 60 * 1000   # no model call may hang the run; a timed-out call is retried  # retrieval calls embeddings; Vertex embedding quota is low


def embed(texts):
    out = []
    for i in range(0, len(texts), 100):
        for attempt in range(8):
            try:
                out += [e.values for e in EMB.models.embed_content(model="text-embedding-005", contents=texts[i:i + 100]).embeddings]
                break
            except Exception as e:
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt + 2))
    return out


RETRIES = {"n": 0, "standard": 0}
LLM_STD = None   # same model on the standard tier, for calls Flex keeps refusing
FLEX_TRIES = int(os.environ.get("FLEX_TRIES", "8"))   # smoke sets 0 to exercise the standard path
DONE = {"n": 0}


def progress(stage, system):
    # a run must never be silent for hours: one log line per 100 finished calls
    DONE["n"] += 1
    if DONE["n"] % 100 == 0:
        log("progress", stage=stage, system=system, done=DONE["n"], retries=RETRIES["n"], spend_sek=round(METER.sek()))


async def gen(model, prompt, max_tokens, cache=None):
    cfg = types.GenerateContentConfig(temperature=0, max_output_tokens=max_tokens, cached_content=cache,
                                      http_options=types.HttpOptions(timeout=CALL_TIMEOUT_MS),
                                      thinking_config=types.ThinkingConfig(thinking_level="LOW"))
    sem = FSEM if cache else SEM
    for attempt in range(20):
        # after FLEX_TRIES refusals the same request goes to the standard tier: same model, same settings,
        # full price (the meter sees the traffic type and prices it so; non-Flex calls are counted and reported)
        client = LLM if attempt < FLEX_TRIES else LLM_STD
        if attempt == FLEX_TRIES:
            RETRIES["standard"] += 1
        try:
            async with sem:
                r = await asyncio.wait_for(client.aio.models.generate_content(model=model, contents=prompt, config=cfg),
                                           CALL_TIMEOUT_MS / 1000 + 60)
            return (r.text or "").strip(), r.usage_metadata.prompt_token_count or 0
        except SystemExit:
            raise
        except Exception as e:
            RETRIES["n"] += 1
            if attempt == 19:
                return f"ERROR {repr(e)[:150]}", 0
            await asyncio.sleep(min(180, 2 ** min(attempt, FLEX_TRIES - 1) + 3) * (0.5 + random.random()))


# ---------------- LoCoMo ----------------
ANSWER_PROMPT = """You answer questions about a long conversation between two people, using only the memory
given below. Dates in session headers tell you when things were said; resolve relative dates
("last week") against them. Answer in a short phrase. If the memory does not contain the answer,
say "not mentioned".

MEMORY:
{context}

QUESTION: {question}
ANSWER:"""
# the same prompt, split where the question starts: the part before it is identical for every question of a
# conversation, so phase D keeps it in an explicit context cache and sends only the rest with each question
ANSWER_HEAD, ANSWER_TAIL = ANSWER_PROMPT[:ANSWER_PROMPT.index("QUESTION:")], ANSWER_PROMPT[ANSWER_PROMPT.index("QUESTION:"):]

JUDGE_PROMPT = """You grade an answer to a question about a conversation.
Question: {question}
Gold answer: {gold}
Generated answer: {answer}
The generated answer is CORRECT if it conveys the same fact as the gold answer, even if it is
shorter, longer or worded differently; for dates, the same day, month or period counts as correct.
Otherwise it is WRONG. Reply with exactly one word: CORRECT or WRONG."""


def sessions(sample):
    conv, out, i = sample["conversation"], [], 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        for t in conv[f"session_{i}"]:
            text = t.get("text", "")
            if t.get("blip_caption"):
                text += f" [shares an image: {t['blip_caption']}]"
            out.append({"id": t.get("dia_id"), "session": i, "date": date, "speaker": t["speaker"], "text": text})
        i += 1
    return out


def render(turns):
    lines, last = [], None
    for t in turns:
        if t["session"] != last:
            lines.append(f"\n--- Session {t['session']} ({t['date']}) ---")
            last = t["session"]
        lines.append(f"[{t['id']}] {t['speaker']}: {t['text']}")
    return "\n".join(lines)


def parse_date(s):
    for f in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return dt.datetime.strptime(s.strip(), f).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            pass
    return dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)


def load(spec):
    import requests
    d = spec["dataset"]
    raw = requests.get(f"https://raw.githubusercontent.com/{d['repo']}/{d['commit']}/{d['file']}", timeout=120).content
    if hashlib.sha256(raw).hexdigest() != d["sha256"]:
        raise SystemExit("LoCoMo sha256 differs from the spec")
    data = json.loads(raw)[:spec.get("n_conversations", N_CONV)]
    if SMOKE:
        data = data[:1]
        data[0] = dict(data[0], qa=[q for q in data[0]["qa"] if q.get("category") != 5][:20])
    return data


def by_session(turns):
    out = {}
    for t in turns:
        out.setdefault(t["session"], []).append(t)
    sess = [out[k] for k in sorted(out)]
    return sess[:3] if SMOKE else sess


# ---------------- systems: ingest(conv) then retrieve(question) -> context ----------------
class Baseline:
    name = "baseline"

    async def prepare(self, cid, turns):
        texts = [f"{t['date']} {t['speaker']}: {t['text']}" for t in turns]
        self.turns = getattr(self, "turns", {})
        self.mem = getattr(self, "mem", {})
        d = tempfile.mkdtemp()
        m = RefMem(os.path.join(d, "m.db"), embed, supersede="append")
        for t, v in zip(texts, embed(texts)):
            m.write("a", t, vector=v)
        self.mem[cid], self.turns[cid] = m, turns

    async def context(self, cid, q):
        got = self.mem[cid].recall("a", q, k=10)
        return render([self.turns[cid][g["id"] - 1] for g in sorted(got, key=lambda g: g["id"])])


class FullContext:
    """Phase D: no memory system; the whole conversation (every session, with its date) is the context.
    The prompt text up to the question goes into one explicit Vertex context cache per conversation; each
    question sends only "QUESTION: ...ANSWER:". A cache that cannot be made stops the run (no silent fallback)."""
    name = "fullcontext"
    TTL_S = 4 * 3600

    async def prepare(self, cid, turns):
        self.ctx, self.cache = getattr(self, "ctx", {}), getattr(self, "cache", {})
        self.ctx[cid] = render(turns)
        head = ANSWER_HEAD.format(context=self.ctx[cid])
        plain = genai.Client(vertexai=True, project=PROJECT, location="global")
        c = await asyncio.to_thread(plain.caches.create, model=ANSWER, config=types.CreateCachedContentConfig(
            contents=[types.Content(role="user", parts=[types.Part(text=head)])], ttl=f"{self.TTL_S}s",
            display_name=f"{RUN_ID[-20:]}-{cid}"))
        self.cache[cid] = c.name
        toks = getattr(c.usage_metadata, "total_token_count", 0) or 0
        # storage is billed per token-hour and is not in any response; charge the whole TTL up front
        METER.extra["fullcontext:cache"] = METER.extra.get("fullcontext:cache", 0.0) + \
            toks / 1e6 * self.TTL_S / 3600 * METER.p["cache_storage_usd_per_million_token_hours"] * METER.p["usd_to_sek"]
        self.cache_tokens = getattr(self, "cache_tokens", {})
        self.cache_tokens[cid] = toks

    async def context(self, cid, q):
        return self.ctx[cid]

    async def cleanup(self):
        plain = genai.Client(vertexai=True, project=PROJECT, location="global")
        for name in getattr(self, "cache", {}).values():
            try:
                await asyncio.to_thread(plain.caches.delete, name=name)
            except Exception as e:
                log("cache delete failed", cache=name, error=repr(e)[:150])


class Mem0Sys:
    name = "mem0"

    def __init__(self):
        from mem0 import Memory
        port = os.environ.get("HS_PG_PORT", "55432")
        pg = os.environ.get("PGDATA_DIR", os.path.expanduser("~/pgdata"))  # on the VM: a host disk, so an ingest survives a crash
        fresh = not os.path.exists(os.path.join(pg, "PG_VERSION"))
        if fresh:
            subprocess.run(["initdb", "-D", pg, "-A", "trust", "-U", "app"], check=True, stdout=subprocess.DEVNULL)
        # the data dir outlives the container; a container that ended without stopping Postgres leaves
        # postmaster.pid behind, and in a fresh container its PID can belong to another process, so pg_ctl
        # refuses to start. Nothing else runs Postgres in this container, so a leftover pid file is stale.
        pidf = os.path.join(pg, "postmaster.pid")
        if os.path.exists(pidf):
            log("removing stale postmaster.pid", content=open(pidf).read()[:80])
            os.remove(pidf)
        r = subprocess.run(["pg_ctl", "-D", pg, "-o", f"-k /tmp -p {port} -c listen_addresses=127.0.0.1 -c max_connections=200",
                            "-l", "/tmp/pg.log", "-w", "start"])
        if r.returncode:
            raise SystemExit("postgres did not start: " + open("/tmp/pg.log").read()[-1500:])
        import atexit   # stop cleanly so the next container starts without recovery
        atexit.register(lambda: subprocess.run(["pg_ctl", "-D", pg, "-m", "fast", "-w", "stop"]))
        if fresh:
            subprocess.run(["createdb", "-h", "127.0.0.1", "-p", port, "-U", "app", "mem0"], check=True)
            subprocess.run(["psql", "-h", "127.0.0.1", "-p", port, "-U", "app", "-d", "mem0", "-c", "CREATE EXTENSION IF NOT EXISTS vector"], check=True)
        cfg = {"llm": {"provider": "gemini", "config": {"model": ANSWER, "vertexai": True, "project": PROJECT,
                                                        "location": "global", "temperature": 0, "max_tokens": 16000}},
               "embedder": {"provider": "gemini", "config": {"model": "text-embedding-005", "embedding_dims": 768, "api_key": "unused"}},
               "vector_store": {"provider": "pgvector", "config": {"dbname": "mem0", "user": "app", "host": "127.0.0.1", "port": int(port),
                                                                   "embedding_model_dims": 768, "collection_name": "locomo", "maxconn": 20}}}
        self.m = Memory.from_config(cfg)
        self.m.llm.client = LLM          # metered, Flex
        self.m.embedding_model.client = EMB
        self.tag = ("smoke" + uuid.uuid4().hex[:6]) if SMOKE else (os.environ.get("MEM0_TAG") or uuid.uuid4().hex[:6])  # same tag on resume, so stored memories are found
        self.failed = 0
        self.outcomes = []               # one entry per session: how its extraction went
        self.tl = threading.local()
        from mem0.memory.utils import remove_code_blocks
        orig = self.m.llm.generate_response

        def spy(*a, **k):
            # Mem0 logs a malformed extraction reply and stores nothing, without raising. Record whether
            # each reply parsed, so a silent drop is never counted as a successful ingest.
            r = orig(*a, **k)
            if (k.get("response_format") or {}).get("type") == "json_object":
                try:
                    ok = isinstance(json.loads(remove_code_blocks(r), strict=False), dict)
                except Exception:
                    ok = False
                self.tl.bad = getattr(self.tl, "bad", False) or not ok
            return r
        self.m.llm.generate_response = spy

    def _add(self, msgs, uid):
        self.tl.bad = False
        r = self.m.add(msgs, user_id=uid)
        return not self.tl.bad, r

    async def _try(self, msgs, uid, tries):
        for attempt in range(tries):
            try:
                ok, _ = await asyncio.to_thread(self._add, msgs, uid)
                if ok:
                    return True, attempt + 1
            except SystemExit:
                raise
            except Exception as e:
                log("mem0 add error", error=repr(e)[:200])
            await asyncio.sleep(min(60, 2 ** attempt + 2))
        return False, tries

    async def prepare(self, cid, turns):
        """Each session is added whole. If Mem0's extraction reply is malformed, the same session is
        retried up to 3 times; if it still fails, it is added in chunks of 8 messages (each retried up to
        3 times). Every session's path is recorded, so the failure rate is itself a measured result."""
        uid = f"{cid}-{self.tag}"
        for n, s in enumerate(by_session(turns)):
            msgs = [{"role": "user", "content": f"[{t['date']}] {t['speaker']}: {t['text']}"} for t in s]
            ok, tries = await self._try(msgs, uid, 3)
            rec = {"conversation": cid, "session": n + 1, "messages": len(msgs), "whole_ok": ok, "whole_tries": tries}
            if not ok:
                chunks = [msgs[i:i + 8] for i in range(0, len(msgs), 8)]
                res = [await self._try(c, uid, 3) for c in chunks]
                rec.update({"chunks": len(chunks), "chunks_failed": sum(1 for r in res if not r[0])})
                self.failed += rec["chunks_failed"] > 0
            self.outcomes.append(rec)

    async def wipe(self, cid):
        await asyncio.to_thread(self.m.delete_all, user_id=f"{cid}-{self.tag}")

    async def stored(self, cid, turns):
        """Count what a reused ingest left in the store, so a lost volume is never scored as empty memory."""
        r = await asyncio.to_thread(self.m.get_all, filters={"user_id": f"{cid}-{self.tag}"}, top_k=100000)
        rows = r.get("results", r) if isinstance(r, dict) else r
        return {"memories": len(rows), "ok": len(rows) > 0}

    async def context(self, cid, q):
        r = await asyncio.to_thread(self.m.search, q, top_k=10, filters={"user_id": f"{cid}-{self.tag}"})
        rows = r.get("results", r) if isinstance(r, dict) else r
        return "\n".join(f"- {x.get('memory', '')}" for x in rows)


class GraphitiSys:
    name = "graphiti"

    def __init__(self):
        self.g, self.failed = None, 0
        self.tag = ("smoke" + uuid.uuid4().hex[:6]) if SMOKE else (os.environ.get("GRAPHITI_TAG") or uuid.uuid4().hex[:6])  # same tag on resume; FalkorDB data is on the VM disk

    async def _g(self):
        if self.g is None:
            from graphiti_core import Graphiti
            from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.gemini_client import GeminiClient
            cfg = LLMConfig(model=ANSWER, small_model=ANSWER, api_key="unused")
            self.g = Graphiti(graph_driver=FalkorDriver(host=os.environ.get("FALKOR_HOST", "127.0.0.1"), port=6379, database=f"locomo_{self.tag}"),
                              llm_client=GeminiClient(config=cfg, client=LLM, max_tokens=16000),
                              embedder=GeminiEmbedder(GeminiEmbedderConfig(embedding_model="text-embedding-005", api_key="unused"), client=EMB),
                              cross_encoder=GeminiRerankerClient(config=cfg, client=LLM))
            await self.g.build_indices_and_constraints()
        return self.g

    async def prepare(self, cid, turns):
        from graphiti_core.nodes import EpisodeType
        g = await self._g()
        for i, s in enumerate(by_session(turns)):
            body = "\n".join(f"{t['speaker']}: {t['text']}" for t in s)
            for attempt in range(8):
                try:
                    await g.add_episode(name=f"{cid}-session-{i + 1}", episode_body=body, source=EpisodeType.message,
                                        source_description=f"conversation session {i + 1}", reference_time=parse_date(s[0]["date"]),
                                        group_id=f"{cid.replace('-', '_')}_{self.tag}")
                    break
                except SystemExit:
                    raise
                except Exception as e:
                    if attempt == 7:
                        self.failed += 1
                        log("graphiti session failed", conversation=cid, error=repr(e)[:200])
                    await asyncio.sleep(min(60, 2 ** attempt + 2))

    async def wipe(self, cid):
        g = await self._g()
        grp = f"{cid.replace('-', '_')}_{self.tag}"
        await g.driver.clone(database=grp).execute_query("MATCH (n) DETACH DELETE n")

    async def stored(self, cid, turns):
        g = await self._g()
        grp = f"{cid.replace('-', '_')}_{self.tag}"
        # graphiti-core 0.31 on FalkorDB keeps each group_id in its own graph, so count inside that graph
        res = await g.driver.clone(database=grp).execute_query("MATCH (e:Episodic {group_id: $g}) RETURN count(e) AS n", g=grp)
        recs = res[0] if isinstance(res, tuple) else res
        n = recs[0]["n"] if recs else 0
        want = len(by_session(turns))
        return {"episodes": n, "sessions": want, "ok": n == want}

    async def context(self, cid, q):
        g = await self._g()
        edges = await g.search(q, group_ids=[f"{cid.replace('-', '_')}_{self.tag}"], num_results=10)

        def when(e):
            a = e.valid_at.strftime("%d %B %Y") if e.valid_at else "unknown date"
            return a + (f" until {e.invalid_at.strftime('%d %B %Y')}" if e.invalid_at else "")
        return "\n".join(f"- {e.fact} ({when(e)})" for e in edges)


# ---------------- run ----------------
def paired_ci(d, n=10000, seed=0):
    rnd = random.Random(seed)
    m = len(d)
    means = sorted(sum(d[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {"mean": round(sum(d) / m, 4), "lo": round(means[int(0.025 * n)], 4), "hi": round(means[int(0.975 * n) - 1], 4), "n_items": m}


async def answer_all(sysobj, samples):
    """All conversations are ingested in parallel (sessions in order within each), then all questions
    are answered in parallel. Flex calls can take a minute each, so parallelism sets the wall time."""
    convs = [(sample.get("sample_id", f"conv{i}"), sessions(sample), sample) for i, sample in enumerate(samples)]
    METER.bucket = f"{sysobj.name}:ingest"
    t0 = time.time()

    done_key = f"ingest_{sysobj.name}.json"
    persistent = sysobj.name in ("mem0", "graphiti")
    reuse = None if SMOKE else SPEC.get("reuse_ingest", {}).get(sysobj.name)
    if persistent and reuse and not runs.blob(f"{BASE}/{done_key}").exists():
        # pre-registered reuse of an earlier ingest made with identical ingest settings (named in the spec)
        b_, _, p_ = reuse["record"][5:].partition("/")
        old = json.loads(gcs.bucket(b_).blob(p_).download_as_bytes())
        if old.get("tag") != getattr(sysobj, "tag", None):
            raise SystemExit(f"{sysobj.name}: reused ingest tag {old.get('tag')} differs from this run's tag {sysobj.tag}")
        put(done_key, {"tag": old["tag"], "reused_from": reuse["record"],
                       "convs": {c: dict(v, reused=True) for c, v in old["convs"].items() if c in reuse["conversations"]}})
    done = json.loads(runs.blob(f"{BASE}/{done_key}").download_as_bytes()) if (
        persistent and runs.blob(f"{BASE}/{done_key}").exists()) else {"tag": getattr(sysobj, "tag", None), "convs": {}}
    lock = asyncio.Lock()

    async def ing(cid, turns):
        if cid in done["convs"] and done.get("tag") == getattr(sysobj, "tag", None):
            chk = await sysobj.stored(cid, turns)
            log("ingest skipped (already stored)", system=sysobj.name, conversation=cid, check=chk)
            if not chk["ok"]:
                raise SystemExit(f"{sysobj.name} {cid}: stored data missing or incomplete {chk}; not scoring")
            async with lock:
                done["convs"][cid]["resume_check"] = chk
                put(done_key, done)
            return
        t = time.time()
        n0 = len(getattr(sysobj, "outcomes", []))
        if persistent:
            # a conversation without a finished-ingest record may hold part of an ingest from a stopped run;
            # it is cleared first, so a stop at any moment can be resumed without duplicate memories
            left = await sysobj.stored(cid, turns)
            if left.get("memories") or left.get("episodes"):
                log("clearing partial ingest before ingesting again", system=sysobj.name, conversation=cid, left=left)
                await sysobj.wipe(cid)
                after = await sysobj.stored(cid, turns)
                if after.get("memories") or after.get("episodes"):
                    raise SystemExit(f"{sysobj.name} {cid}: partial ingest could not be cleared {after}")
        await sysobj.prepare(cid, turns)
        chk = (await sysobj.stored(cid, turns)) if persistent else None   # every fresh ingest is checked too
        if chk and not chk["ok"]:
            raise SystemExit(f"{sysobj.name} {cid}: ingest left incomplete data {chk}; not scoring")
        async with lock:
            done["convs"][cid] = {"seconds": round(time.time() - t), "ingest_check": chk,
                                  "outcomes": [o for o in getattr(sysobj, "outcomes", [])[n0:] if o["conversation"] == cid]}
            if persistent:
                put(done_key, done)
        log("ingest done", system=sysobj.name, conversation=cid, seconds=round(time.time() - t), spend_sek=round(METER.sek()))
    await asyncio.gather(*(ing(cid, turns) for cid, turns, _ in convs))
    ingest_s = time.time() - t0
    put("state_meter.json", {"|".join(k): v for k, v in METER.by.items()})
    METER.bucket = f"{sysobj.name}:answer"

    async def one(cid, i, qa):
        ctx = None
        for attempt in range(10):
            try:
                async with RSEM:
                    ctx = await sysobj.context(cid, qa["question"])
                break
            except SystemExit:
                raise
            except Exception as e:
                if attempt == 9:
                    raise
                await asyncio.sleep(min(90, 2 ** attempt + 3))
        if sysobj.name == "fullcontext":
            ans, tin = await gen(ANSWER, ANSWER_TAIL.format(question=qa["question"]), 2048, cache=sysobj.cache[cid])
        else:
            ans, tin = await gen(ANSWER, ANSWER_PROMPT.format(context=ctx, question=qa["question"]), 2048)
        progress("answer", sysobj.name)
        return {"system": sysobj.name, "conversation": cid, "q_index": i, "category": qa.get("category"),
                "question": qa["question"], "gold": str(qa["answer"]), "answer": ans, "prompt_tokens": tin,
                "context_chars": len(ctx)}
    jobs = [(cid, i, qa) for cid, _, sample in convs
            for i, qa in enumerate(q for q in sample["qa"] if q.get("category") != 5 and "answer" in q)]
    # answers are saved as they come, so a stopped run keeps every good answer and a resumed run asks only
    # the questions still missing; an answer that failed is asked again, never scored
    pkey = f"answers_{sysobj.name}.partial.json"
    got = {}
    if runs.blob(f"{BASE}/{pkey}").exists():
        got = {(r["conversation"], r["q_index"]): r for r in json.loads(runs.blob(f"{BASE}/{pkey}").download_as_bytes())
               if not r["answer"].startswith("ERROR")}
        log("answers resumed", system=sysobj.name, kept=len(got), of=len(jobs))
    plock = asyncio.Lock()

    async def keep(j):
        r = await one(*j)
        async with plock:
            if not r["answer"].startswith("ERROR"):
                got[(j[0], j[1])] = r
                if len(got) % 50 == 0:
                    put(pkey, list(got.values()))
        return r
    for p in range(3):   # the first pass, then up to two more for answers that failed
        todo = [j for j in jobs if (j[0], j[1]) not in got]
        if not todo:
            break
        if p:
            log("asking failed answers again", system=sysobj.name, n=len(todo), pass_=p)
        await asyncio.gather(*(keep(j) for j in todo))
        put(pkey, list(got.values()))
    missing = [j for j in jobs if (j[0], j[1]) not in got]
    if missing:
        raise SystemExit(f"{sysobj.name}: {len(missing)} of {len(jobs)} answers still failed after 3 passes; "
                         "good answers are saved, rerun to finish")
    rows = [got[(j[0], j[1])] for j in jobs]
    if hasattr(sysobj, "cleanup"):
        await sysobj.cleanup()
    log("answers done", system=sysobj.name, questions=len(rows), ingest_s=round(ingest_s), spend_sek=round(METER.sek()))
    return list(rows)


async def judge(rows, model):
    METER.bucket = f"judge:{model}"

    async def one(r):
        v, _ = await gen(model, JUDGE_PROMPT.format(question=r["question"], gold=r["gold"], answer=r["answer"]), 512)
        progress("judge", r["system"])
        return v.upper()
    return await asyncio.gather(*(one(r) for r in rows))


async def main():
    global METER, LLM, EMB, SPEC
    import concurrent.futures
    asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=64))
    b, _, p = os.environ["SPEC_URI"][5:].partition("/")
    raw = gcs.bucket(b).blob(p).download_as_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != os.environ.get("EXPECTED_SPEC_SHA"):
        raise SystemExit("spec sha256 differs; refusing to run")
    spec = SPEC = json.loads(raw)
    if spec.get("judge_policy") != "pro_all":
        raise SystemExit("this runner implements judge_policy pro_all only")
    METER = Meter(spec["price_assumptions_usd_per_million_tokens_before_flex_discount"], spec["cost_guard_sek"])
    # the spec's guard can only be raised by Mohamed's budget approval for this run, stored in the run folder
    if runs.blob(f"{BASE}/budget_approval.json").exists():
        ap = json.loads(runs.blob(f"{BASE}/budget_approval.json").download_as_bytes())
        if ap.get("run_id") != RUN_ID or not ap.get("approved_by") or not ap.get("quote"):
            raise SystemExit("budget_approval.json is not a valid approval for this run")
        METER.cap = float(ap["guard_sek"])
        log("cost guard raised by approval", guard_sek=METER.cap, round_cap_sek=ap["round_cap_sek"], at=ap["approved_at"])
    LLM, EMB = clients()
    global LLM_STD
    LLM_STD = metered(genai.Client(vertexai=True, project=PROJECT, location="global"))
    if not runs.blob(f"{BASE}/plan.json").exists():
        put("plan.json", {"run_id": RUN_ID, "bench": spec["bench"], "spec_sha256": sha, "spec_uri": os.environ["SPEC_URI"],
                          "written_at": dt.datetime.now(dt.timezone.utc).isoformat(), "service_account": "narva-tester"})
    prior = {}
    if runs.blob(f"{BASE}/state.json").exists():
        prior = json.loads(runs.blob(f"{BASE}/state.json").download_as_bytes())
        METER.by = {tuple(k.split("|", 1)): v for k, v in prior.get("meter", {}).items()}
        METER.extra = prior.get("meter_extra", {})
    gate = f"smoke-gates/{sha[:16]}-{CODE_SHA[:16]}-{'-'.join(sorted(SYSTEMS))}.ok"
    if not SMOKE and not runs.blob(gate).exists():
        raise SystemExit(f"no passed smoke run for this spec, code and systems ({gate}); run with SMOKE=1 first")
    samples = load(spec)
    judge_model = prior.get("judge_model")
    results = prior.get("systems", {})
    for name in SYSTEMS:
        if name in results:   # resumed run: a finished system is never answered or judged again
            log("system already done", system=name, accuracy=results[name]["accuracy"])
            continue
        sysobj = {"baseline": Baseline, "fullcontext": FullContext, "mem0": Mem0Sys, "graphiti": GraphitiSys}[name]()
        log("system start", system=name)
        rows = await answer_all(sysobj, samples)
        judge_model = PRO  # spec v2.0: gemini-3.1-pro-preview judges every answer; no light judge
        errs = sum(1 for r in rows if r["answer"].startswith("ERROR"))
        if errs > 0.02 * len(rows):
            raise SystemExit(f"{name}: {errs} of {len(rows)} answers failed; not scoring a broken run")
        verdicts = await judge(rows, judge_model)
        for r, v in zip(rows, verdicts):
            r["judge"], r["correct"] = v, v.startswith("CORRECT")
        put(f"answers_{name}.json", rows)
        results[name] = {"accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4), "questions": len(rows),
                         "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in rows) / len(rows)),
                         "failed_ingest_sessions": getattr(sysobj, "failed", 0),
                         "cache_tokens": getattr(sysobj, "cache_tokens", None),
                         "cached_input_tokens": sum(v[4] for (b, m), v in METER.by.items() if b == f"{name}:answer"),
                         "non_flex_calls": sum(v[3] for (b, m), v in METER.by.items() if b.startswith(f"{name}:")),
                         "ingest_outcomes": [o for v in (json.loads(runs.blob(f"{BASE}/ingest_{name}.json").download_as_bytes())["convs"].values()
                                                         if name != "baseline" and runs.blob(f"{BASE}/ingest_{name}.json").exists() else [])
                                             for o in v["outcomes"]] or None,
                         "unparsed_judgements": sum(1 for r in rows if not r["judge"].startswith(("CORRECT", "WRONG"))),
                         "answer_errors": sum(1 for r in rows if r["answer"].startswith("ERROR")),
                         "by_conversation": {c: round(sum(r["correct"] for r in rows if r["conversation"] == c) /
                                                      max(1, sum(1 for r in rows if r["conversation"] == c)), 4)
                                             for c in sorted({r["conversation"] for r in rows})},
                         "by_category": {str(c): round(sum(r["correct"] for r in rows if r["category"] == c) /
                                                       max(1, sum(1 for r in rows if r["category"] == c)), 4) for c in (1, 2, 3, 4)},
                         "spend_sek": {"ingest": round(METER.sek(f"{name}:ingest"), 1), "answer": round(METER.sek(f"{name}:answer"), 1)}}
        put("state.json", {"judge_model": judge_model, "systems": results, "meter": {"|".join(k): v for k, v in METER.by.items()},
                           "meter_extra": METER.extra})
        log("system done", system=name, result=results[name], spend_sek_total=round(METER.sek()))
    # comparisons against the baseline, by code
    comp = {}
    have_base = runs.blob(f"{BASE}/answers_baseline.json").exists()
    base = {(r["conversation"], r["q_index"]): r for r in json.loads(runs.blob(f"{BASE}/answers_baseline.json").download_as_bytes())} if have_base else {}
    for name in (results if have_base else []):
        if name == "baseline":
            continue
        rs = json.loads(runs.blob(f"{BASE}/answers_{name}.json").download_as_bytes())
        ci = paired_ci([r["correct"] - base[(r["conversation"], r["q_index"])]["correct"] for r in rs])
        comp[name] = {"diff_vs_baseline": ci, "verdict": "Better than baseline" if ci["lo"] > 0 else
                      "Worse than baseline" if ci["hi"] < 0 else "No difference"}
    # phase D: every memory system and the baseline against the whole conversation in the context window
    comp_fc = {}
    if runs.blob(f"{BASE}/answers_fullcontext.json").exists():
        fc = {(r["conversation"], r["q_index"]): r for r in json.loads(runs.blob(f"{BASE}/answers_fullcontext.json").download_as_bytes())}
        for name in results:
            if name == "fullcontext":
                continue
            rs = json.loads(runs.blob(f"{BASE}/answers_{name}.json").download_as_bytes())
            ci = paired_ci([r["correct"] - fc[(r["conversation"], r["q_index"])]["correct"] for r in rs])
            comp_fc[name] = {"diff_vs_fullcontext": ci, "verdict": "Better than full context" if ci["lo"] > 0 else
                             "Worse than full context" if ci["hi"] < 0 else "No difference"}
    put("summary.json", {"run_id": RUN_ID, "bench": spec["bench"], "spec_sha256": sha, "judge_model": judge_model,
                         "systems": results, "comparison": comp, "comparison_vs_fullcontext": comp_fc, "meter": METER.report(),
                         "measured_spend_sek": round(METER.sek(), 1), "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    log("phase B finished", measured_spend_sek=round(METER.sek(), 1), comparison=comp)
    if SMOKE:
        bad = [n for n, r in results.items() if r["answer_errors"] or r["unparsed_judgements"] or r["failed_ingest_sessions"]
               or (n == "fullcontext" and not r["cached_input_tokens"])]   # the cache must really be used
        if bad:
            raise SystemExit(f"smoke run found problems in {bad}; no gate written")
        runs.blob(gate).upload_from_string(json.dumps({"smoke_run": RUN_ID, "spec_sha256": sha, "code_sha256": CODE_SHA,
                                                       "systems": SYSTEMS, "spend_sek": round(METER.sek(), 1)}))
        log("smoke passed; gate written", gate=gate)


if __name__ == "__main__":
    asyncio.run(main())
