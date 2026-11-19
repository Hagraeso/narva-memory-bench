# Does agent memory beat plain retrieval?

An independent, pre-registered comparison of agent memory systems on the whole LoCoMo benchmark, by Narva.

**Read the write-up with figures:** https://hagraeso.github.io/narva-memory-bench/

We ran four ways of giving an answer model a long conversation through one protocol on all of LoCoMo (10 conversations, 1,540 questions in categories 1 to 4). Every answer was graded by the same stronger model.

| System | Correct | Gap to vector search | 95% interval | Prompt tokens per question |
|---|---:|---:|---|---:|
| Whole conversation in the context window | **79.9%** | +19.2 | +16.9 to +21.6 | 26,751 |
| Vector search over raw turns (10 turns) | 60.7% | baseline | | 681 |
| Graphiti 0.31.2 (FalkorDB) | 52.1% | −8.5 | −11.3 to −5.7 | 387 |
| Mem0 open source 2.2.1 | 45.8% | −14.9 | −17.9 to −11.9 | 475 |

## Findings

1. **If the history fits in the context window, put all of it in.** The largest prompt was 30,892 tokens. Whole-conversation answering beat retrieval by 19.2 points, Graphiti by 27.7 and Mem0 by 34.1, at about 39 times the prompt tokens of retrieval.
2. **Extracting facts at write time lost information.** Both memory systems trailed a plain vector search over the raw turns, and both answered "not mentioned" more often (Mem0 470, Graphiti 412, vector search 347, whole conversation 83).
3. **Mem0 open source dates old conversations by the day of ingest.** In mem0ai 2.2.1 the extraction prompt treats its Observation Date as the only temporal anchor, and the open-source `add()` never sets it, so it is always today. 82 of Mem0's 321 time answers named 2026, the year we ingested the data; the conversations took place in 2022 and 2023. Time accuracy: Mem0 38.6%, vector search 63.5%.

These agree with earlier reports that simple approaches do well on LoCoMo ([Letta](https://www.letta.com/blog/benchmarking-ai-agent-memory); the Mem0 paper's own full-context row). What this adds is an independent run of the whole set under one pre-registered protocol, with paired intervals, one strong judge for every answer, a full-context baseline, and the date defect with its cause.

## By question type

| Question type (LoCoMo category) | Questions | Whole conversation | Vector search | Graphiti | Mem0 |
|---|---:|---:|---:|---:|---:|
| Linking several facts (1) | 282 | 64.5% | 38.3% | 36.5% | 39.7% |
| Time (2) | 321 | 80.7% | 63.5% | 65.7% | 38.6% |
| Open-domain inference (3) | 96 | 40.6% | 19.8% | 21.9% | 22.9% |
| Single fact (4) | 841 | 89.2% | 71.7% | 55.6% | 53.1% |

## Method

- **Pre-registered spec:** [`spec/DOM-01-phaseB-v2.1.json`](spec/DOM-01-phaseB-v2.1.json), sha256 `fc83e666b10da9a1ffe26ce9e7ab8817bfba4e734525b03f47c3e85a8f6fa547`. It fixed models, prompts, statistics and verdict rules before the run. A full run was refused by code until a smoke run with the same spec and code had passed.
- **Same everything:** one answer prompt and one answer model (`gemini-3-flash-preview`, low thinking, temperature 0) for all four systems; each memory system used the same model internally and `text-embedding-005` for embeddings. Every answer was graded by `gemini-3.1-pro-preview` against the LoCoMo answer key.
- **Systems:** Mem0 open source 2.2.1, each session added whole with its date inside the message text (the open-source SDK refuses a timestamp argument), top 10 memories. Graphiti 0.31.2 on FalkorDB, each session one episode with its date as reference time, top 10 facts with their valid dates. Vector search: every raw turn embedded, top 10 turns. Whole conversation: every session in order with its date, through an explicit Vertex AI context cache.
- **Statistics:** paired difference per question, 95% percentile bootstrap, 10,000 resamples, seed 0. "Better" or "worse" needs the whole interval on one side of zero.
- **Disclosed repairs and reuse** (all in [`results/`](results)): 18 grades came back empty and were asked again with the same judge and a larger output cap (9 then correct). Ingest of the first 3 conversations for Mem0 and Graphiti was reused from earlier runs with identical settings, after a check that the stored data was complete. The run was stopped and resumed four times; each stop, its cause and the code change are recorded in `resume_note_*.json`. No failed answer was scored.
- **Cost:** about 616 SEK of metered model calls on Google's Flex tier.

## Limits

- One dataset, English only. Every conversation fit in the context window, so this says nothing about histories that do not. That is where a memory system has to earn its place, and it is the next test.
- Two memory systems, both open source. Mem0's hosted platform and newer published algorithm were not tested.
- Each system ran under this shared protocol, not its vendor's own evaluation recipe.

## Rerun it on your own data

[`kit/`](kit) holds `narva-assay`: one command reruns a finding on your own conversations under this protocol (same prompts, judge rule, baseline and statistics), with your memory system plugged in, and reports whether the finding held on your data. Nothing is spent until you pass `--yes`. See [`kit/README.md`](kit/README.md).

## Repository

| Path | What |
|---|---|
| `spec/` | The pre-registered spec |
| `code/phaseB.py` | The runner (final version; earlier versions differ only in the resilience changes listed in the resume notes) |
| `code/repair_verdicts.py` | Re-asks empty judge verdicts only |
| `code/refmem.py` | The vector-search baseline's text rendering and store |
| `code/Dockerfile`, `code/requirements.txt`, `code/vm_round1.sh` | Image and VM script used for the run (Google Cloud specific) |
| `results/summary.json` | Accuracy, categories, conversations, intervals, metered spend |
| `results/answers_*.json` | Every question, answer key, answer, verdict and prompt size, per system |
| `kit/` | `narva-assay`, the rerun kit (MIT) |
| `docs/index.html` | The write-up |

The same results are on Hugging Face: https://huggingface.co/datasets/Hagras87/narva-memory-bench-locomo

## Licenses

- Code: MIT (see `LICENSE`).
- Results in `results/` contain LoCoMo questions and answer keys, so they are shared under **CC BY-NC 4.0**, the license of [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024, "Evaluating Very Long-Term Conversational Memory of LLM Agents").

## About Narva

Narva is an independent test bench for agent memory. It sells no memory system. Corrections are welcome as issues; if you think a system was set up unfairly, say how and we will rerun it under a new pre-registered spec.
