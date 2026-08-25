# quera-mzi-harness

Agent harness for extracting V_pi and V_null per stage from
ScreeningDatasetV2 (QuEra Photonics Data Engineer Intern screening,
Q3 + Q4). Python 3.11+, numpy + scipy only; `anthropic` needed only for
live runs.

## Pipeline (terminal view)

```
 dataset ----read-only,---> +------------- agent sandbox ---------------+
 (npz sweeps,      jailed   | loop.py: sonnet-4-6 live/mock, 1 JSON/turn |
  captures,                 |     |                                      |
  lab logs)                 | Dispatcher: schema . path jail . budgets --+--> trajectory log
                            |     |                     |                |    (append-only,
                            |   tools:              skills (RO):         |     hash-chained)
                            |   fit_fringe          SKILL.md ->          |         |
                            |   (multi-start)       reference.md         |         v
                            +--------------------------------------------+  evals + verifiers
                                          |                                 5/5 . 7/7 . 5/5
                                          v                                 read the log, never
                              final JSON answer + Q1 figure                 the self-report
```

## Layout
```
agent/
  tools.py        deterministic tools: fit_fringe (multi-start model
                  comparison), list_dataset, read_lab_log; tool registry
  guardrails.py   the out-of-reach layer: schema validation, dataset path
                  jail, call/time budgets, append-only hash-chained
                  trajectory log
  loop.py         the agentic loop: system prompt assembly (skill INDEX +
                  exact final-answer schema), MockClient (offline) and
                  AnthropicClient (live), one-JSON-per-turn protocol with
                  bounded repair, run()
skills/mzi-fringe-analysis/
  SKILL.md        progressive disclosure L1 (frontmatter index) + L2 (body)
  reference.md    L3: model math, degeneracies, tolerances - loaded only
                  on demand
evals/
  verifiers.py    deterministic verifiers (no model): synthetic recovery,
                  model discrimination, schema, determinism, error paths
  eval_vnull.py   behavioural eval, happy path: asserts on the trajectory
                  (skill-before-fit, provenance, caveat, budgets), never
                  on the agent's self-report
  eval_badnpz.py  behavioural eval, failure path: drift-corrupted sweep
                  must trigger the skill's step 6 - refuse to report
                  parameters, consult a lab log (the dataset's "RAG
                  seam"), and ground the refusal in the real chi2
runs/             hash-chained trajectories; live1.jsonl is a recorded
                  claude-sonnet-4-6 run on the real dataset
make_q1_plot.py   regenerates q1_static_fit.png from fit_fringe output
                  (single source of truth for the Q1 figure)
sample_dataset/   synthetic stand-in with the ScreeningDatasetV2 schema
```

## Quickstart (offline, no API key)
```
pip install -r requirements.txt
python evals/verifiers.py                 # 5 deterministic verifiers
python evals/eval_vnull.py sample_dataset # happy path: 7 behavioural checks
python evals/eval_badnpz.py               # failure path: 5 checks
```

## Live runs (proof of the model using the skill)
```
set ANTHROPIC_API_KEY=...          (export on POSIX)
python -m agent.loop <dataset_root> --live --name live1
python evals/eval_vnull.py <dataset_root> --live
python evals/eval_badnpz.py --live
```
Against a live claude-sonnet-4-6 on the real dataset, both behavioural
evals pass in full (7/7 and 5/5); `runs/live1.jsonl` is the recorded
trajectory.

## Design decisions (details in the write-up)
- **No framework, on purpose.** LangGraph/CrewAI hide exactly the parts
  this exercise assesses (loop, context assembly, guardrails); the Claude
  Agent SDK is excellent but manages context for you, which Q3.2 asks to
  be defended explicitly. ~450 lines keeps every harness part inspectable.
- **System prompt gets**: role, tool schemas, skill index (name +
  description lines), and an exact final-answer schema. **Not**: skill
  bodies, raw arrays, lab logs, past trajectories - pulled on demand.
- **Numbers never originate in the model.** All numeric work happens in
  tools; evals check the final answer equals tool output bit-for-bit.
- **Out of reach**: dispatcher, budgets, and the hash-chained trajectory
  log sit outside the tool surface; evals read the log, not the agent's
  account of itself.
- The synthetic-recovery verifier caught a real bug during development
  (single-start curve_fit in a false minimum), which is why fit_fringe
  multi-starts.
- First live-model contact exposed two more harness gaps the mock could
  not: an underspecified final-answer schema (the model invented its own
  key names) and a fatal single-violation protocol path (the model
  narrated a plan and batched tool calls). Fixed in loop.py - exact
  schema, first-object JSON parse, bounded repair - with the evals kept
  frozen as the spec that caught both.

## Storage note
Storage is deliberately abstracted behind the dataset root: in production,
datasets, manifests, and the hash-chained trajectories would land in Azure
Blob with the same schemas and provenance fields, matching QuEra's
Microsoft stack. Nothing in the harness assumes local disk.

## AI-use notes (honest log)
AI assistance was used throughout, per the assessment policy. Three places
it had to be overridden or where verification caught it out: (1) an
AI-suggested inspection script named `inspect.py` shadowed Python's stdlib
`inspect` module and crashed numpy's import chain - diagnosed and renamed;
(2) the first fit implementation used single-start `curve_fit`, which the
synthetic-recovery verifier caught converging to a false minimum
(V_pi 4.3/30 V instead of 6.4/9.2 V) - replaced with multi-start; (3) an
AI-drafted capture risetime analysis produced artifact numbers on
large-amplitude (fringe-wrapping) waveforms and was discarded rather than
reported; only the small-signal waveform-identity check (r = 0.993)
survived verification. The checks catching the tools is the design
working as intended.
