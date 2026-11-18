"""Repair empty judge verdicts in a finished phase B run. No answer is regenerated.

In run phaseB-20261117T175124Z-r10e92, gemini-3.1-pro-preview returned an empty verdict for 18 answers
(the 512-token output cap was spent on thinking), and the runner scored those answers as wrong. This
job asks the same judge (same model, prompt, temperature 0, thinking LOW, Flex) again for exactly those
answers, with a 2,048-token output cap, then recomputes accuracy and the paired intervals with the
runner's own code. Originals are kept as *.prerepair.json. The run stops if any verdict stays unparsed.
Env: RUN_ID, SPEC_URI, CAP_SEK (default 10).
"""
import asyncio
import json
import os

import phaseB as P

OK = ("CORRECT", "WRONG")


def get(name):
    return json.loads(P.runs.blob(f"{P.BASE}/{name}").download_as_bytes())


def score(rows):
    r = {"accuracy": round(sum(x["correct"] for x in rows) / len(rows), 4), "questions": len(rows),
         "unparsed_judgements": sum(1 for x in rows if not x["judge"].startswith(OK)),
         "by_conversation": {c: round(sum(x["correct"] for x in rows if x["conversation"] == c) /
                                      sum(1 for x in rows if x["conversation"] == c), 4)
                             for c in sorted({x["conversation"] for x in rows})},
         "by_category": {str(c): round(sum(x["correct"] for x in rows if x["category"] == c) /
                                       sum(1 for x in rows if x["category"] == c), 4)
                         for c in sorted({x["category"] for x in rows})}}
    return r


async def main():
    b, _, p = os.environ["SPEC_URI"][5:].partition("/")
    spec = json.loads(P.gcs.bucket(b).blob(p).download_as_bytes())
    P.METER = P.Meter(spec["price_assumptions_usd_per_million_tokens_before_flex_discount"], float(os.environ.get("CAP_SEK", "10")))
    P.METER.bucket = "repair:judge"
    P.LLM, P.EMB = P.clients()
    P.LLM_STD = P.metered(P.genai.Client(vertexai=True, project=P.PROJECT, location="global"))
    summary = get("summary.json")
    if "verdict_repair" in summary:
        raise SystemExit("this run was already repaired")
    answers = {n: get(f"answers_{n}.json") for n in summary["systems"]}
    fixed = {}
    for name, rows in answers.items():
        bad = [r for r in rows if not r["judge"].startswith(OK)]
        if not bad:
            continue
        P.put(f"answers_{name}.prerepair.json", rows)
        for r in bad:
            for _ in range(3):
                v, _ = await P.gen(P.PRO, P.JUDGE_PROMPT.format(question=r["question"], gold=r["gold"], answer=r["answer"]), 2048)
                if v.upper().startswith(OK):
                    break
            r["judge"], r["correct"], r["judge_repaired"] = v.upper(), v.upper().startswith("CORRECT"), True
        still = [r for r in bad if not r["judge"].startswith(OK)]
        if still:
            raise SystemExit(f"{name}: {len(still)} verdicts still unparsed; nothing written")
        fixed[name] = {"repaired": len(bad), "now_correct": sum(r["correct"] for r in bad)}
        P.log("verdicts repaired", system=name, **fixed[name])
    if not fixed:
        raise SystemExit("no unparsed verdicts")
    P.put("summary.prerepair.json", summary)
    for name, rows in answers.items():
        if name in fixed:
            P.put(f"answers_{name}.json", rows)
            summary["systems"][name].update(score(rows))
    key = lambda r: (r["conversation"], r["q_index"])
    base = {key(r): r for r in answers["baseline"]}
    fc = {key(r): r for r in answers["fullcontext"]}
    for name, rows in answers.items():
        if name != "baseline":
            ci = P.paired_ci([r["correct"] - base[key(r)]["correct"] for r in rows])
            summary["comparison"][name] = {"diff_vs_baseline": ci, "verdict": "Better than baseline" if ci["lo"] > 0 else
                                           "Worse than baseline" if ci["hi"] < 0 else "No difference"}
        if name != "fullcontext":
            ci = P.paired_ci([r["correct"] - fc[key(r)]["correct"] for r in rows])
            summary["comparison_vs_fullcontext"][name] = {"diff_vs_fullcontext": ci, "verdict": "Better than full context" if ci["lo"] > 0 else
                                                          "Worse than full context" if ci["hi"] < 0 else "No difference"}
    summary["verdict_repair"] = {"why": "empty verdicts from the judge (output cap spent on thinking) had been scored as wrong",
                                 "how": "same judge model, prompt, temperature and thinking level, output cap 2048 instead of 512, only for those answers",
                                 "systems": fixed, "spend_sek": round(P.METER.sek(), 2), "code_sha256_prefix": P.CODE_SHA[:16]}
    summary["measured_spend_sek"] = round(summary["measured_spend_sek"] + P.METER.sek(), 1)
    P.put("summary.json", summary)
    P.log("repair done", **{n: summary["systems"][n]["accuracy"] for n in fixed})


if __name__ == "__main__":
    asyncio.run(main())
