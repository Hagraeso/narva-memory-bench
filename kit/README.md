# narva-assay

Rerun a Narva store item's memory test on your own conversations, under the same protocol Narva used.

Narva measured memory systems for AI agents on LoCoMo (round 1, bench DOM-01-phaseB-v2.1) and keeps the results that change a builder's decision as store items. Each item states a prediction and the condition that would break it. This kit lets you check that prediction on your data in one command.

## What it covers

| Store item | What the run compares | The prediction is broken if |
|---|---|---|
| DOM-01-S17 | Whole conversation in the context window against Narva's vector baseline | Any memory system's interval against the whole history is entirely above zero on histories that fit the window |
| DOM-01-S16 | Your memory system against the baseline | Your system's interval against plain retrieval is entirely above zero |
| DOM-01-S18 | Your memory system's time answers | No time answer names a year that is in none of the conversation's session dates |
| DOM-01-S5 | Keyword plus vector retrieval against vector only | The hybrid's interval is entirely above zero |
| DOM-01-S6 | Two passes of the same configuration | Both passes give the same verdict on every question |

Not covered yet: the multi-agent items (S2, S3, S7 to S12), which need each product's server.

## Install

```
pip install ./kit            # or: pip install "./kit[gemini]" for Gemini
```

## Run

```
narva-assay list
narva-assay estimate DOM-01-S17 --data locomo10.json
narva-assay run DOM-01-S17 --data locomo10.json --backend gemini \
    --model gemini-3-flash-preview --judge gemini-3.1-pro-preview --yes
```

Without `--yes` the run prints a token estimate and stops, so nothing is spent until you confirm. Answers are saved every 50 questions; a stopped run resumes and asks only what is missing. `narva-assay score DIR` rebuilds the report from saved answers with no model calls.

Backends: `gemini` (Vertex AI when `GOOGLE_CLOUD_PROJECT` is set, otherwise `GEMINI_API_KEY`), `openai` (any OpenAI-compatible endpoint: `OPENAI_BASE_URL`, `OPENAI_API_KEY`), and `fake` (no network, for checking your adapter).

## Your data

LoCoMo's JSON works as is. Your own conversations go in JSONL, one conversation per line:

```
{"id": "c1", "sessions": [{"date": "2:15 pm on 3 May, 2024", "turns": [{"speaker": "Ana", "text": "..."}]}],
 "questions": [{"question": "When did Ana move?", "answer": "April 2024", "time": true}]}
```

Mark questions about time with `"time": true`; S18 reads only those.

## Your memory system

Subclass `MemorySystem` with `prepare(cid, turns)` and `context(cid, question)`, then pass `--system your_module:YourClass`. See `examples/my_memory.py`.

## Comparable to Narva only when the setup matches

Same prompts, judge rule, baseline and statistics as Narva (a test checks the prompts byte for byte and reproduces round 1's intervals from its saved answers). Your numbers are comparable to Narva's only with the same answer model, judge model and embedding model: Narva used gemini-3-flash-preview (low thinking), gemini-3.1-pro-preview and text-embedding-005 on Vertex AI. text-embedding-005 is not on the Gemini API; there, pass another `--embed-model` and treat the baseline as yours, not Narva's.

The report says whether Narva's prediction held, broke or was inconclusive on your data. It is your run, not Narva evidence; send `report.json` with the answers files to have it considered for Narva's prediction record.

## Protocol

Answer and judge prompts, the baseline (each turn one memory with its session date, top 10 turns by cosine similarity, shown in conversation order), temperature 0, and a paired bootstrap of 10,000 resamples with seed 0. Verdict: Works if the interval is entirely above zero, Harms if entirely below, otherwise No effect. An empty judge verdict is asked again with a larger output cap; a failed answer is asked again and never scored.

## Checking the kit itself

```
python3 -I kit/tests/test_kit.py locomo10.json
```

With no model and no money, this checks that the prompts and the baseline store match Narva's runner byte for byte, reproduces round 1's intervals from `results/answers_*.json`, checks the LoCoMo file against the pre-registered sha256, and runs every recipe end to end with the fake model, including a stop and resume.

Code: MIT.
