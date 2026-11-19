"""Checks that need no model and no money. Run: python3 -I tests/test_kit.py <locomo10.json> [round1_dir]
round1_dir holds Narva's round 1 answers files (answers_<system>.json with a 'correct' field per question); in the
public repository it defaults to ../results. Works in the public repository layout (code/phaseB.py, code/refmem.py)
and in Narva's internal one (tester/phaseB.py, bench/refmem.py)."""
import ast
import json
import os
import shutil
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.dirname(KIT)
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "examples"))
from narva_assay import cli, core  # noqa: E402

LOCOMO = sys.argv[1]
R1 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CODE, "results")


def first(*paths):
    return next(p for p in paths if os.path.exists(p))


PHASEB = first(os.path.join(CODE, "code", "phaseB.py"), os.path.join(CODE, "tester", "phaseB.py"))
REFMEM = first(os.path.join(CODE, "code", "refmem.py"), os.path.join(CODE, "bench", "refmem.py"))
ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail else ""))


# 1. the prompts and the reference memory are Narva's own, byte for byte
tree = ast.parse(open(PHASEB).read())
consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
          if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
check("answer prompt identical to Narva's runner", consts["ANSWER_PROMPT"] == core.ANSWER_PROMPT)
check("judge prompt identical to Narva's runner", consts["JUDGE_PROMPT"] == core.JUDGE_PROMPT)
check("refmem identical to Narva's baseline store", open(REFMEM).read() == open(os.path.join(KIT, "narva_assay", "refmem.py")).read())

# 2. the statistics reproduce round 1's published intervals from round 1's own answers
A = {n: {(r["conversation"], r["q_index"]): r for r in json.load(open(os.path.join(R1, f"answers_{n}.json")))}
     for n in ("baseline", "fullcontext", "mem0", "graphiti")}
want = {"fullcontext": (0.192, 0.169, 0.216), "mem0": (-0.149, -0.179, -0.119), "graphiti": (-0.085, -0.113, -0.057)}
for n, (m, lo, hi) in want.items():
    c = core.compare(A[n], A["baseline"])
    got = (round(c["mean"], 3), round(c["lo"], 3), round(c["hi"], 3))
    check(f"round 1 interval {n} vs baseline", all(abs(g - w) <= 0.0006 for g, w in zip(got, (m, lo, hi))), f"{got} vs published {(m, lo, hi)}")
check("round 1 accuracy baseline 60.7%", round(core.accuracy(A["baseline"]), 3) == 0.607)

# 3. the data loader sees what Narva saw: 10 conversations, 1,540 questions, 321 time questions
convs, sha = core.load_data(LOCOMO)
check("LoCoMo file is the pre-registered one", sha == "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4")
check("1,540 questions in 10 conversations", (len(convs), sum(len(q) for _, _, q in convs)) == (10, 1540))
check("321 time questions", sum(q["time"] for _, _, qs in convs for q in qs) == 321)
check("largest conversation renders to over 100,000 characters (round 1 largest prompt 30,892 tokens)",
      max(len(core.render(t)) for _, t, _ in convs) > 100000)

# 4. end to end with the fake model on 1 conversation: every recipe, a stop and resume, and the score command
out = tempfile.mkdtemp()
for item, extra in (("DOM-01-S17", []), ("DOM-01-S5", []), ("DOM-01-S6", []),
                    ("DOM-01-S16", ["--system", "my_memory:KeepLastTurns"]), ("DOM-01-S18", ["--system", "my_memory:KeepLastTurns"])):
    d = os.path.join(out, item)
    res = cli.main(["run", item, "--data", LOCOMO, "--limit", "1", "--backend", "fake", "--model", "fake", "--judge", "fake",
                    "--out", d, "--yes"] + extra)
    check(f"{item} runs end to end", res and res["outcome_on_this_data"] in ("held", "broken", "inconclusive"),
          res and res["outcome_on_this_data"])

d = os.path.join(out, "DOM-01-S17")
f = os.path.join(d, "answers_baseline.json")
rows = json.load(open(f))
json.dump(rows[: len(rows) // 2], open(f, "w"))
calls = {"n": 0}
orig = core.Fake.generate


def counted(self, prompt, mt):
    calls["n"] += 1
    return orig(self, prompt, mt)


core.Fake.generate = counted
cli.main(["run", "DOM-01-S17", "--data", LOCOMO, "--limit", "1", "--backend", "fake", "--model", "fake", "--judge", "fake", "--out", d, "--yes"])
missing = len(rows) - len(rows) // 2
check("a resumed run asks only the missing questions", calls["n"] == 2 * missing, f"{calls['n']} calls for {missing} missing answers")
check("resumed answers are complete", len(json.load(open(f))) == len(rows))
core.Fake.generate = orig
r = cli.report("DOM-01-S17", d, json.load(open(os.path.join(d, "meta.json"))), core.load_data(LOCOMO, 1)[0])
check("score rebuilds the report without calls", r["systems"]["baseline"]["questions"] == len(rows))
check("run without --yes spends nothing", cli.main(["run", "DOM-01-S17", "--data", LOCOMO, "--limit", "1", "--backend", "fake",
                                                    "--model", "fake", "--judge", "fake", "--out", os.path.join(out, "no")]) is None
      and not os.path.exists(os.path.join(out, "no")))

# 5. recipes carry the store's own prediction wording
items_file = os.path.join(CODE, "tools", "value_items_DOM-01.json")
if os.path.exists(items_file):   # internal layout only: the store's item file is not in the public repository
    store = {s["id"]: s for s in json.load(open(items_file))["items"]}
    check("recipes match the store's predictions", all(v["prediction"] == store[k]["prediction"]["text"] for k, v in cli.ASSAYS.items()))
shutil.rmtree(out)
print(f"\n{sum(ok)} of {len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
