"""narva-assay: rerun a Narva store item's test on your own conversations.

  narva-assay list
  narva-assay estimate ITEM --data FILE [--limit N]
  narva-assay run ITEM --data FILE --backend gemini|openai|fake --model M --judge J [--system module:Class]
                  [--embed-model E] [--out DIR] [--limit N] [--parallel 8] [--flex] --yes
  narva-assay score DIR        rebuild the report from saved answers, no model calls

Without --yes, run prints the estimate and stops: nothing is spent until you confirm.
"""
import argparse
import asyncio
import json
import os
import sys

from . import core

HERE = os.path.dirname(__file__)
ASSAYS = json.load(open(os.path.join(HERE, "assays.json")))


def rule(expr, values):
    """A rule like 'lo>0' or 'hi<0' or 'lo<=0': one field, one comparison, one number (no eval)."""
    import operator
    import re
    m = re.fullmatch(r"\s*(\w+)\s*(<=|>=|<|>|==)\s*(-?[\d.]+)\s*", expr)
    if not m:
        raise ValueError(f"bad rule {expr!r}")
    ops = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge, "==": operator.eq}
    return ops[m.group(2)](values[m.group(1)], float(m.group(3)))


def outcome(item, res):
    """Is this run consistent with the item's prediction? held / broken / inconclusive, by the rule in assays.json."""
    a = ASSAYS[item]
    kind = a["check"]
    if kind == "compare":
        c = res["comparison"]
        if rule(a["broken_if"], c):
            return "broken"
        return "held" if rule(a["held_if"], c) else "inconclusive"
    if kind == "foreign_years":
        return "held" if res["foreign_years"]["count"] > 0 else "broken"
    if kind == "agreement":
        return "held" if res["agreement"]["share"] < 1 else "broken"
    raise ValueError(kind)


def systems_for(item, system_spec):
    a = ASSAYS[item]
    if a.get("needs_system") and not system_spec:
        raise SystemExit(f"{item} tests a memory system: pass --system your_module:YourClass (see README)")
    return [system_spec if s == "system" else s for s in a["systems"]]


def report(item, out_dir, meta, convs):
    a = ASSAYS[item]
    rows = {}
    for label in meta["labels"]:
        rows[label] = core._load(os.path.join(out_dir, f"answers_{label}.json"))
    res = {"item": item, "title": a["title"], "prediction": a["prediction"], "data": meta["data"], "data_sha256": meta["data_sha256"],
           "models": meta["models"], "systems": {}, "made_at": core.now(), "protocol": "Narva DOM-01-phaseB-v2.1 (answer prompt, judge prompt, baseline, paired bootstrap 10,000 resamples seed 0)"}
    for label, r in rows.items():
        res["systems"][label] = {"accuracy": core.accuracy(r), "questions": len(r),
                                 "prompt_tokens_mean": round(sum(x["prompt_tokens"] for x in r.values()) / len(r)),
                                 "prompt_tokens_max": max(x["prompt_tokens"] for x in r.values()),
                                 "not_mentioned": sum("not mentioned" in x["answer"].lower() for x in r.values())}
    labels = meta["labels"]
    if a["check"] == "compare":
        res["comparison"] = dict(core.compare(rows[labels[0]], rows[labels[1]]), a=labels[0], b=labels[1])
    elif a["check"] == "foreign_years":
        by = {cid: turns for cid, turns, _ in convs}
        n, of = core.foreign_years(rows[labels[0]], by)
        res["foreign_years"] = {"count": n, "time_answers": of}
    elif a["check"] == "agreement":
        share, n = core.agreement(rows[labels[0]], rows[labels[1]])
        res["agreement"] = {"share": share, "questions": n}
    res["outcome_on_this_data"] = outcome(item, res)
    json.dump(res, open(os.path.join(out_dir, "report.json"), "w"), ensure_ascii=False, indent=1)
    md = [f"# {item}: {a['title']}", "", f"Data: {meta['data']} (sha256 {meta['data_sha256'][:12]}…). Models: {meta['models']}.", "",
          "| System | Accuracy | Questions | Mean prompt tokens | Largest prompt | 'not mentioned' |", "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {k} | {v['accuracy']:.1%} | {v['questions']} | {v['prompt_tokens_mean']:,} | {v['prompt_tokens_max']:,} | {v['not_mentioned']} |" for k, v in res["systems"].items()]
    if "comparison" in res:
        c = res["comparison"]
        md += ["", f"{c['a']} minus {c['b']}: {100 * c['mean']:+.1f} points, 95% interval {100 * c['lo']:+.1f} to {100 * c['hi']:+.1f} "
                   f"({c['n_items']} paired questions). Verdict: **{c['verdict']}**."]
    if "foreign_years" in res:
        f = res["foreign_years"]
        md += ["", f"{f['count']} of {f['time_answers']} time answers name a year that appears in no session date of their conversation."]
    if "agreement" in res:
        g = res["agreement"]
        md += ["", f"Two passes of the same configuration gave the same verdict on {g['share']:.1%} of {g['questions']} questions."]
    md += ["", f"Narva's prediction: {a['prediction']}", f"On this data the prediction **{res['outcome_on_this_data']}**.", "",
           a.get("note", ""), "", "This is your run, not Narva evidence. To have it count in Narva's prediction record, send report.json and the answers files."]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(md) + "\n")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(prog="narva-assay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("estimate", "run"):
        s = sub.add_parser(name)
        s.add_argument("item")
        s.add_argument("--data", required=True)
        s.add_argument("--limit", type=int)
        s.add_argument("--system")
        if name == "run":
            s.add_argument("--backend", choices=["gemini", "openai", "fake"], required=True)
            s.add_argument("--model", required=True)
            s.add_argument("--judge", required=True)
            s.add_argument("--embed-model")
            s.add_argument("--out")
            s.add_argument("--parallel", type=int, default=8)
            s.add_argument("--flex", action="store_true")
            s.add_argument("--yes", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("dir")
    a = p.parse_args(argv)

    if a.cmd == "list":
        for k, v in ASSAYS.items():
            print(f"{k}: {v['title']}\n    compares: {', '.join(v['systems'])}")
        print("\nNot in this kit yet: the multi-agent items (DOM-01-S2, S3, S7 to S12); they need each product's server.")
        return
    if a.cmd == "score":
        meta = json.load(open(os.path.join(a.dir, "meta.json")))
        convs, _ = core.load_data(meta["data"], meta.get("limit"))
        print(json.dumps(report(meta["item"], a.dir, meta, convs), indent=1))
        return
    if a.item not in ASSAYS:
        raise SystemExit(f"unknown item {a.item}; see: narva-assay list")
    convs, sha = core.load_data(a.data, a.limit)
    sysnames = systems_for(a.item, a.system)
    est = core.estimate(convs, [s.split("#")[0] for s in sysnames], getattr(a, "model", ""))
    print(json.dumps({"item": a.item, "conversations": len(convs), "estimate": est,
                      "judge_calls": sum(e["questions"] for e in est.values())}, indent=1))
    if a.cmd == "estimate" or not a.yes:
        if a.cmd == "run":
            print("Nothing was run. Check the estimate against your model's prices, then add --yes.")
        return
    kw = {"embed_model": a.embed_model} if a.embed_model else {}
    if a.backend == "gemini":
        kw["flex"] = a.flex
    answer = core.make_model(a.backend, a.model, **kw)
    judge = core.make_model(a.backend, a.judge, **kw)
    out = a.out or f"assay-{a.item}-{core.now().replace(':', '')}"
    os.makedirs(out, exist_ok=True)
    labels = []
    for s in sysnames:
        name, _, rep = s.partition("#")
        sysobj = core.make_system(name, answer)
        label = sysobj.name + (f"#{rep}" if rep else "")
        labels.append(label)
        asyncio.run(core.answer_and_judge(sysobj, convs, answer, judge, out, a.parallel, label))
    meta = {"item": a.item, "data": os.path.abspath(a.data), "data_sha256": sha, "limit": a.limit, "labels": labels,
            "models": {"backend": a.backend, "answer": a.model, "judge": a.judge, "embed": a.embed_model or "default", "flex": a.flex}}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    res = report(a.item, out, meta, convs)
    print(open(os.path.join(out, "report.md")).read())
    return res


if __name__ == "__main__":
    sys.exit(main() and 0)
