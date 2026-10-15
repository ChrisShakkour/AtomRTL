# AtomRTL

An RTL-generation agent built on LangGraph and a local LLM (Ollama), or Claude over the Anthropic API
(see [Running on Claude](#running-on-claude-anthropic-api)). Given a spec, it writes
SystemVerilog and checks it with Icarus Verilog, slang and an interface check, fixing errors in a loop.

## How it works

```
parse_spec → plan → generate → write_code → check → route ─┬─ no errors / out of budget → finalize
   [py]      [LLM]   [LLM]       [py]        [py]    [py]  ├─ errors → fix [LLM] → write_code
                                                           └─ same errors twice → generate again
```

- **Python nodes** do everything deterministic: parse the module name and ports from the spec, write
  the file, compile with iverilog, lint with slang, check the interface, and decide the next step.
- **LLM nodes** (`plan`, `generate`, `fix`) have no tools. Each gets a small, fresh prompt and replies
  with one code block.
- Settings (model, temperatures per node, attempt budget) are in `atomrtl/config.py` and can be
  overridden with a YAML file passed as `--config`.

## Simulation feedback (optional)

Off by default. When enabled, a design that passes the static checks is also simulated against a
reference testbench, and an LLM node fixes the mismatches:

```
route ─ clean → simulate [py] → route_sim [py] ─┬─ pass / out of sim budget / stuck → finalize
                                                └─ mismatches → sim_fix [LLM] → write_code → check ...
```

- `simulate` compiles the testbench + reference + design with Icarus (same as scoring) in `<workdir>/sim/`,
  reads the per-output `Hint:` lines and the testbench's `wave.vcd`.
- `sim_fix` is one LLM call with a fresh prompt: spec, current code, the per-output mismatch summary, and a
  table of the inputs and failing outputs at each clock edge around the first mismatch (expected and
  actual values). It never sees the testbench or reference source. It replies with a short diagnosis
  and the full corrected code, which goes back through the static checks (and `fix` if it broke the build).
- A sim fix that increases the mismatches is discarded, and the next fix starts from the best version.
  A fix that changes nothing gets a "the cause is elsewhere" note in the next prompt; after `stuck_after`
  identical results in a row (default 3) the loop stops.
  When the agent gives up, the best simulated code is what's left in `<module>.sv`.
- Settings under `simulation:` in the config: `enabled`, `max_fixes` (default 4, separate from
  `max_attempts`), `show_expected` (set `false` to hide the expected values from the table),
  `cycles_before`/`cycles_after`/`max_rows` (table window), `stuck_after`, `timeout`.

```bash
# One spec
python -m atomrtl.cli --spec ../verilog-eval/dataset_spec-to-rtl/Prob100_fsm3comb_prompt.txt --top TopModule \
    --sim-tb ../verilog-eval/dataset_spec-to-rtl/Prob100_fsm3comb_test.sv \
    --sim-ref ../verilog-eval/dataset_spec-to-rtl/Prob100_fsm3comb_ref.sv -v
# Benchmark
python scripts/run_verilog_eval.py --sim
```

On verilog-eval this uses the **same testbench that scores the result**, so it is oracle feedback: a
pass rate with simulation is not comparable to one without it, and is reported separately.

## Usage

```bash
# One spec → runs/<timestamp>/<spec-name>/
python -m atomrtl.cli --spec ../verilog-eval/dataset_spec-to-rtl/Prob011_norgate_prompt.txt -v

# verilog-eval benchmark (agent, then scoring against the hidden testbench)
python scripts/run_verilog_eval.py --problems 'Prob0[0-2]*'      # or no --problems for all 156
python scripts/run_verilog_eval.py --resume-dir runs/<timestamp>  # finish an interrupted run
python scripts/run_verilog_eval.py --sim                          # with simulation feedback (see below)

# Tests (no LLM needed)
pytest
```

Each problem directory contains the generated `TopModule.sv`, `result.json` (agent outcome),
`transcript.log` / `transcript.jsonl` (every LLM call and response, every tool command and its
output, every routing decision), and for benchmark runs `validation.json` / `validation.log`
(scoring) and `status.json`. The run directory has `command.txt`, `config.yaml` and `summary.json`.

## One-shot baseline (`--mode oneshot`)

A run without the agent, to measure what the graph adds. Each problem gets one LLM call, and the reply is
written to the file as-is:

```
parse_spec [py] → oneshot [LLM] → write_code [py] → finalize_oneshot [py]
```

- There is no plan, no iverilog/slang/interface check, no fix and no simulation. `--sim` is rejected.
- `--oneshot-prompt` (or `oneshot_prompt:` in the config) picks the prompt:
  - `agent` (default): the agent's own `generate` prompt with no plan section: the system rules, the
    spec, the interface header parsed from it, and the single-code-block instruction. Against the full agent,
    this measures what the plan and the check/fix/sim loop add.
  - `verilog_eval`: verilog-eval's own spec-to-rtl prompt (`scripts/sv-generate`). The system message is
    "You are a Verilog RTL designer that only writes code using correct Verilog syntax.", and the user
    message is the spec verbatim. Results compare with published verilog-eval numbers, and against
    `agent` this measures what AtomRTL's prompt adds.
- The code is the last fenced block containing `endmodule`. With no fence, the text from the first `module`
  to the last `endmodule` is used.
- `result.json` says `unchecked` (code written) or `no_code`. Only the benchmark's scoring decides pass/fail,
  so `agent_self_check_pass` in the summary is always 0.

```bash
python scripts/run_verilog_eval.py --mode oneshot --config configs/claude.yaml   # or --config configs/ollama.yaml
python scripts/run_verilog_eval.py --mode oneshot --oneshot-prompt verilog_eval --config configs/claude.yaml
```

## Running on Claude (Anthropic API)

Ollama stays the default. `provider: anthropic` (or `--provider anthropic`) sends every LLM node to
Claude Sonnet 5.5 (`claude-sonnet-5-5`) instead. No server is started, and nothing else in the graph changes.

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # from console.anthropic.com; never put it in a config file
export ANTHROPIC_WORKSPACE_ID=wrkspc_... # only if the key isn't scoped to a workspace (400 "not scoped to a workspace")

python -m atomrtl.cli --spec ../verilog-eval/dataset_spec-to-rtl/Prob011_norgate_prompt.txt --provider anthropic -v
python scripts/run_verilog_eval.py --config configs/claude.yaml --limit 5   # try a few before all 156
python scripts/run_verilog_eval.py --config configs/ollama.yaml             # back to the local model
```

- Settings are under `anthropic:` in the config: `model`, `effort` (`low`…`max`, default `medium`),
  `max_tokens` (default 16000), `num_parallel` (default `--jobs`, 8), `max_retries`, `timeout` and
  `refusal_fallback`. `--model` overrides the model of whichever provider is active.
- Claude Sonnet 5.5 always thinks (adaptive) and rejects sampling settings, so `temperature` and
  `nodes.*.temperature` are ignored on this provider. `max_tokens` covers thinking plus the reply, which is
  why `simulation.max_tokens` (an Ollama loop guard) doesn't apply here. `nodes.<node>.max_tokens` still does.
- `refusal_fallback` (on by default) lets the API rerun a request that a safety classifier declined on a
  fallback model. A reply that still ends in `stop=refusal` shows up in the transcript and goes through the
  normal "no code block" path.
- Cost: a full verilog-eval run with the local model used ~0.8M input and ~0.3M output tokens. At $2/$10 per
  million tokens, plus thinking, expect roughly $10–20 per full run. Token totals are in `summary.json`.

## Results vs. TigressRTL

Complete runs against the full 156-problem verilog-eval spec-to-rtl set, all with the same dataset and
scoring. The first five columns use the same model (`devstral-small-2` via Ollama); the last one is
AtomRTL on Claude Sonnet 5.5 (see "Running on Claude"). The "+ sim" columns let the agent
simulate against the scoring testbench (oracle feedback, see "Simulation feedback (optional)"), so compare
them only with each other, not with the no-simulation columns.

| | AtomRTL | TigressRTL | AtomRTL + sim (expected shown) | AtomRTL + sim (expected hidden) | TigressRTL + sim | AtomRTL + sim, Claude Sonnet 5.5 |
|---|---|---|---|---|---|---|
| Run | `2026-10-05_23-15-16` | `2026-08-10_11-40-31` | `2026-10-10_22-27-40` | `2026-10-11_00-00-12` | `2026-09-26_17-30-29` | `2026-10-12_23-40-04` |
| Pass rate | 84/156 (53.8%) | 81/156 (51.9%) | **105/156 (67.3%)** | **105/156 (67.3%)** | 88/156 (56.4%) | **149/156 (95.5%)** |
| Build failures | 2 | 16 | 1 | 1 | 9 | 0 |
| Functional failures | 66 | 57 | 47 | 47 | 56 | 4 |
| Other (sim timeout, etc.) | 4 | 2 | 3 | 3 | 3 | 3 |
| Passed only after a sim fix | - | - | 22 | 22 | - | 5 |
| Total tokens | 497,605 | 3,467,554 | 1,136,337 | 1,023,290 | 29,084,828 | 583,597 |
| Avg tokens/problem | ~3,190 | ~22,228 | ~7,284 | ~6,560 | ~186,441 | ~3,741 |
| Total compute time | ~5,143s (1.4h) | ~7,988s (2.2h) | ~10,936s (3.0h) | ~9,810s (2.7h) | ~29,192s (8.1h) | ~2,131s (0.6h) |
| Avg time/problem | 33.0s | 51.2s | 70.1s | 62.9s | 187.1s | 13.7s |

TigressRTL is a single ReAct agent with tool access (`write_file`, `build_verilog`,
`run_simulation`, ...) and one continuously-growing conversation for the whole problem — every
retry resends the full history so far, not just the new turn. AtomRTL's `plan`/`generate`/`fix`/`sim_fix`
nodes have no tools and each gets a small, fresh, independent prompt (see "How it works" above),
with Python code (not the model) deciding what happens next. That architectural difference is
most of why AtomRTL uses roughly 7x fewer tokens per problem than TigressRTL without simulation,
and 26-28x fewer with it — a handful of TigressRTL's hardest problems individually spent
400K-800K tokens on repeated fix-and-resimulate cycles that never converged.

With simulation, AtomRTL passes 17 more problems than TigressRTL (105 vs. 88) in about a third of the
time. Simulation is worth +21 problems to AtomRTL (53.8% to 67.3%) for ~2.1-2.3x the tokens, against
+7 for TigressRTL (51.9% to 56.4%) for ~8.4x the tokens. The two simulation setups don't give the model
identical feedback: TigressRTL's run predates two fixes (its VCD debug context was leaking the
reference module's values, and there was no fallback for when the model diagnoses a fix correctly but
fails to apply it), and a clean 156-problem TigressRTL rerun with the current code doesn't exist yet.

### On Claude Sonnet 5.5

The same graph and prompts on Claude Sonnet 5.5 (`effort: medium`, 8 problems in parallel) pass 149/156,
with about half the tokens of the devstral simulation runs and 1.06 attempts per problem on average. The
run cost about $2.30 in API usage and took about 5 minutes of wall-clock time. Of the 7 failures:

- 3 can't be passed by any design (`Prob082_lfsr32`, `Prob141_count_clock`,
  `Prob156_review2015_fancytimer`). Their testbench needs about 2M ps of stimulus but prints `TIMEOUT` at
  1M ps, and the reference design times out against its own testbench too. Every run in the table fails
  them, so the passable ceiling is 153, and Claude passed 149 of those (97.4%).
- 3 are an AtomRTL issue (`Prob034_dff8`, `Prob053_m2014_q4d`, `Prob104_mt2015_muxdff`). The reference
  powers up at 0, which the spec doesn't say. `sim_fix` adds an `initial` block, the
  `MultipleAlwaysAssigns` lint check rejects it, and `fix` deletes it again until the loop gives up.
- 1 is a real miss (`Prob066_edgecapture`, 4/266 mismatches).

### What the agent adds

Two one-shot baselines (`--mode oneshot`, see "One-shot baseline") against the full agent, on the same model
and scoring.

**devstral-small-2 (Ollama, 2 problems in parallel)**

| | One-shot, verilog-eval prompt | One-shot, agent prompt (no plan) | Full agent | Full agent + sim |
|---|---|---|---|---|
| Run | `2026-10-16_01-23-49` | `2026-10-16_01-47-34` | `2026-10-05_23-15-16` | `2026-10-10_22-27-40` |
| Pass rate | 53/156 (34.0%) | 70/156 (44.9%) | 84/156 (53.8%) | **105/156 (67.3%)** |
| Build failures | 39 | 16 | 2 | 1 |
| Functional failures | 62 | 67 | 66 | 47 |
| Other (sim timeout, etc.) | 2 | 3 | 4 | 3 |
| LLM calls | 156 | 156 | 353 | 599 |
| Total tokens | 107,520 | 149,142 | 497,605 | 1,136,337 |
| Avg tokens/problem | ~689 | ~956 | ~3,190 | ~7,284 |
| Avg time/problem | 17.1s | 11.9s | 33.0s | 70.1s |

For devstral, each layer adds problems. AtomRTL nearly doubles the plain verilog-eval baseline (53 to 105):

- **Prompt: +17.** The build failures drop from 39 to 16. With the plain prompt, devstral writes invalid
  Verilog: output `wire`s assigned in `always` blocks, `genvar` inside `always`, enum types Icarus rejects.
  The agent's system rules target exactly these.
- **Plan + check/fix loop: +14.** The remaining build failures almost all go away (16 to 2).
- **Simulation feedback: +21.** It's the only layer that cuts functional failures (66 to 47).

The one-shot verilog-eval column has only 2 "other": one of the 3 untestable problems failed to compile
first. Its time per problem includes loading the model onto the GPU (~155s on the first two problems).

**Claude Sonnet 5.5 (Anthropic API, 8 problems in parallel)**

There is no Claude "full agent without simulation" run yet.

| | One-shot, verilog-eval prompt | One-shot, agent prompt (no plan) | Full agent + sim |
|---|---|---|---|
| Run | `2026-10-16_01-12-35` | `2026-10-16_01-16-21` | `2026-10-12_23-40-04` |
| Pass rate | 145/156 (92.9%) | 143/156 (91.7%) | **149/156 (95.5%)** |
| Build failures | 2 | 0 | 0 |
| Functional failures | 6 | 10 | 4 |
| Other (sim timeout, etc.) | 3 | 3 | 3 |
| LLM calls | 156 | 156 | 346 |
| Total tokens | 109,000 | 207,827 | 583,597 |
| Avg tokens/problem | ~699 | ~1,332 | ~3,741 |
| API cost | ~$0.70 | ~$0.87 | ~$2.34 |
| Avg time/problem | 7.8s | 7.9s | 13.7s |

Claude gets much less from the scaffolding than devstral (one-shot is already 92-93%):

- The prompt alone makes no measurable difference: 145 vs. 143, within run-to-run variance (see
  "Variance between runs"). But the two runs fail different problems:
  - The agent prompt removes the build failures: the verilog-eval prompt's 2 named the module
    `top_module`/`full_module` instead of `TopModule`.
  - The agent prompt's strict "use this header exactly" rule backfires on spec typos. `Prob031_dff`'s spec
    lists `input q`, so the model kept `q` as an input and failed 118/121 samples. With the verilog-eval
    prompt it made `q` an output and passed.
- The plan and the check/fix/sim loop add 6 problems over the same prompt (143 to 149) for ~2.8x the
  tokens. The full agent fails only the 3 untestable problems and 4 others.
- 2 compiles in the verilog-eval-prompt run (`Prob045_edgedetect2`, `Prob046_dff8p`) were killed by
  the scoring's 60s timeout during the run, although the designs compile instantly. They were re-scored
  afterwards with the same code (both pass), and `status.json` notes it.

### Variance between runs

Showing or hiding the expected values made no measurable difference to AtomRTL: identical totals, but
12 problems passed only with them shown and 12 different ones only with them hidden. Every problem the
no-simulation baseline passed was passed by at least one of the two simulation runs. That spread
(24 of 156 problems flipping between two runs of nearly the same configuration) is run-to-run variance,
so a single run can't resolve differences smaller than several problems; small subsets (e.g. 11
problems) are noise.

## Environment

Uses conda (conda-forge) for Python and system tools, and pip for Python packages.

- `environment.yml`: Python version, conda packages, and the pip requirements file
- `requirements.txt`: Python packages, pinned with `==`

### Fresh machine setup (Linux)

```bash
# 1. Install Miniforge (conda with conda-forge as the default channel)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
~/miniforge3/bin/conda init bash && source ~/.bashrc

# 2. Clone the repo and create the environment
git clone https://github.com/ChrisShakkour/AtomRTL.git
cd AtomRTL
conda env create -f environment.yml
conda activate atomrtl

# 3. Install Ollama (LLM server, runs outside conda)
curl -fsSL https://ollama.com/install.sh | sh            # with sudo
# or, without sudo, install to ~/.local/ollama:
mkdir -p ~/.local/ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x -C ~/.local/ollama
echo 'export PATH="$HOME/.local/ollama/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# 4. Download the model (the model name is set in atomrtl/config.py)
ollama serve &          # only needed for this pull, skip if a server is already running
ollama pull devstral-small-2

# 5. Benchmark dataset (optional), cloned next to AtomRTL
cd .. && git clone https://github.com/NVlabs/verilog-eval.git && cd AtomRTL
```

Scripts start the Ollama server themselves via `atomrtl.llm.ollama_server()` if it isn't running,
so there is no need to run `ollama serve` before using them.

### Updating

```bash
# After editing either file
conda env update -f environment.yml --prune
```

Add every dependency to the files above, never install with a one-off `pip install`.
