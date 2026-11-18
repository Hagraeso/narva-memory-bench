"""narva-ref-mem: Narva's small reference memory service for the DOM-01 bench.

Each pilot idea sits behind one switch, so the bench can run it off and on with everything else fixed:

  I1 event_time    off: the latest ARRIVAL is current      on: the latest OBSERVED time is current
  I2 server_scope  off: scope is a tag filter the caller   on: the server enforces each memory's scope
                        may pass or omit                        from the caller's identity on every read
  I3 hybrid        off: vector ranking only                on: reciprocal rank fusion of BM25 + vector
  I4 supersede     "append": keep every write active (no invalidation)
                   "overwrite": delete the old value (no history)
                   "soft": mark the old value superseded, keep it in history (the idea)

Storage is SQLite (WAL, one connection per writer thread) with an FTS5 table for keywords.
Vectors are held in memory. Conflicts are keyed by an explicit key supplied by the writer;
deriving keys from free text is out of scope for this bench.
"""
import math
import sqlite3
import threading

RRF_K = 60
CANDIDATES = 50


class RefMem:
    def __init__(self, path, embed, event_time=False, server_scope=False, hybrid=False, supersede="soft"):
        assert supersede in ("append", "overwrite", "soft")
        self.path, self.embed = path, embed
        self.event_time, self.server_scope, self.hybrid, self.supersede = event_time, server_scope, hybrid, supersede
        self.vec = {}
        self.vlock = threading.Lock()
        self.local = threading.local()
        db = self._db()
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS mem(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, key TEXT,
              owner TEXT, scope TEXT, tags TEXT, observed_at REAL, status TEXT DEFAULT 'active',
              superseded_by INTEGER);
            CREATE INDEX IF NOT EXISTS mem_key ON mem(key);
            CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(text);
        """)
        db.commit()

    def _db(self):
        if not hasattr(self.local, "db"):
            self.local.db = sqlite3.connect(self.path, timeout=60, isolation_level=None, check_same_thread=False)
        return self.local.db

    # ---------------- write ----------------
    def write(self, owner, text, scope="shared", tags=(), key=None, observed_at=0.0, vector=None):
        v = vector if vector is not None else self.embed([text])[0]
        db = self._db()
        db.execute("BEGIN IMMEDIATE")
        try:
            if key is not None and self.supersede == "overwrite":
                old = [r[0] for r in db.execute("SELECT id FROM mem WHERE key=?", (key,))]
                db.executemany("DELETE FROM mem WHERE id=?", [(i,) for i in old])
                db.executemany("DELETE FROM mem_fts WHERE rowid=?", [(i,) for i in old])
            cur = db.execute("INSERT INTO mem(text,key,owner,scope,tags,observed_at) VALUES(?,?,?,?,?,?)",
                             (text, key, owner, scope, ",".join(tags), observed_at))
            new_id = cur.lastrowid
            db.execute("INSERT INTO mem_fts(rowid,text) VALUES(?,?)", (new_id, text))
            if key is not None and self.supersede == "soft":
                rows = list(db.execute("SELECT id,observed_at FROM mem WHERE key=? AND status='active'", (key,)))
                if self.event_time:  # I1 on: the latest observed time wins; ties go to the later arrival
                    keep = max(rows, key=lambda r: (r[1], r[0]))[0]
                else:                # I1 off: the latest arrival wins
                    keep = new_id
                db.executemany("UPDATE mem SET status='superseded', superseded_by=? WHERE id=?",
                               [(keep, r[0]) for r in rows if r[0] != keep])
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        with self.vlock:
            self.vec[new_id] = (v, math.sqrt(sum(x * x for x in v)) or 1.0)
        return new_id

    # ---------------- read ----------------
    def _visible(self, caller, tags):
        """Row filter as SQL. caller=None means a request with no agent identity."""
        if self.server_scope:  # I2 on: the server decides from identity; caller tags cannot widen it
            if caller is None:
                return "scope='shared'", ()
            return "(scope='shared' OR owner=?)", (caller,)
        if tags:               # I2 off: only the tags the caller chose to pass
            return "(" + " OR ".join("(','||tags||',') LIKE ?" for _ in tags) + ")", tuple(f"%,{t},%" for t in tags)
        return "1=1", ()

    def recall(self, caller, query, k=10, tags=None, qvec=None):
        where, args = self._visible(caller, tags)
        if self.supersede == "soft":
            where += " AND status='active'"
        db = self._db()
        allowed = {r[0] for r in db.execute(f"SELECT id FROM mem WHERE {where}", args)}
        if not allowed:
            return []
        q = qvec if qvec is not None else self.embed([query])[0]
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        with self.vlock:
            items = [(i, self.vec[i]) for i in allowed if i in self.vec]
        vs = sorted(((sum(a * b for a, b in zip(q, v)) / (qn * n), i) for i, (v, n) in items), reverse=True)
        if not self.hybrid:
            ranked = [i for _, i in vs[:k]]
        else:
            terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if len(t) > 1]
            bm = []
            if terms:
                fts_q = " OR ".join(f'"{t}"' for t in terms)
                bm = [r[0] for r in db.execute(
                    "SELECT rowid FROM mem_fts WHERE mem_fts MATCH ? ORDER BY bm25(mem_fts) LIMIT ?",
                    (fts_q, CANDIDATES * 20)) if r[0] in allowed][:CANDIDATES]
            score = {}
            for rank, i in enumerate([i for _, i in vs[:CANDIDATES]]):
                score[i] = score.get(i, 0) + 1 / (RRF_K + rank + 1)
            for rank, i in enumerate(bm):
                score[i] = score.get(i, 0) + 1 / (RRF_K + rank + 1)
            ranked = [i for i, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))[:k]]
        rows = {r[0]: r for r in db.execute(
            f"SELECT id,text,key,owner,scope,observed_at,status FROM mem WHERE id IN ({','.join('?' * len(ranked))})",
            ranked)} if ranked else {}
        return [dict(zip(("id", "text", "key", "owner", "scope", "observed_at", "status"), rows[i])) for i in ranked]

    def history(self, key):
        return [dict(zip(("id", "text", "observed_at", "status"), r)) for r in self._db().execute(
            "SELECT id,text,observed_at,status FROM mem WHERE key=? ORDER BY observed_at", (key,))]
