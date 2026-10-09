# AtomRTL

An RTL-generation agent built on LangGraph and a local LLM (Ollama). Given a spec, it writes
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

## Usage

```bash
# One spec → runs/<timestamp>/<spec-name>/
python -m atomrtl.cli --spec ../verilog-eval/dataset_spec-to-rtl/Prob011_norgate_prompt.txt -v

# verilog-eval benchmark (agent, then scoring against the hidden testbench)
python scripts/run_verilog_eval.py --problems 'Prob0[0-2]*'      # or no --problems for all 156
python scripts/run_verilog_eval.py --resume-dir runs/<timestamp>  # finish an interrupted run

# Tests (no LLM needed)
pytest
```

Each problem directory contains the generated `TopModule.sv`, `result.json` (agent outcome),
`transcript.log` / `transcript.jsonl` (every LLM call and response, every tool command and its
output, every routing decision), and for benchmark runs `validation.json` / `validation.log`
(scoring) and `status.json`. The run directory has `command.txt`, `config.yaml` and `summary.json`.

## Results vs. TigressRTL

Three complete runs against the full 156-problem verilog-eval spec-to-rtl set, all on the same
model (`devstral-small-2` via Ollama) and the same dataset/scoring:

| | AtomRTL | TigressRTL (no simulation) | TigressRTL (with simulation) |
|---|---|---|---|
| Run | `2026-10-05_23-15-16` | `2026-08-10_11-40-31` | `2026-09-26_17-30-29` |
| Pass rate | 84/156 (53.8%) | 81/156 (51.9%) | 88/156 (56.4%) |
| Build failures | 2 | 16 | 9 |
| Functional failures | 66 | 57 | 56 |
| Other (sim timeout, etc.) | 4 | 2 | 3 |
| Total tokens | 497,605 | 3,467,554 | 29,084,828 |
| Avg tokens/problem | ~3,190 | ~22,228 | ~186,441 |
| Total compute time | ~5,148s (1.4h) | ~7,988s (2.2h) | ~29,192s (8.1h) |
| Avg time/problem | 33.0s | 51.2s | 187.1s |

TigressRTL is a single ReAct agent with tool access (`write_file`, `build_verilog`,
`run_simulation`, ...) and one continuously-growing conversation for the whole problem — every
retry resends the full history so far, not just the new turn. AtomRTL's `plan`/`generate`/`fix`
nodes have no tools and each gets a small, fresh, independent prompt (see "How it works" above),
with Python code (not the model) deciding what happens next. That architectural difference is
most of why AtomRTL uses roughly 7x fewer tokens per problem than even TigressRTL's no-simulation
baseline, and nearly 60x fewer than its with-simulation run — a handful of TigressRTL's hardest
problems individually spent 400K-800K tokens on repeated fix-and-resimulate cycles that never
converged.

Simulation measurably helps TigressRTL's pass rate (51.9% to 56.4%) and roughly halves its build
failures (16 to 9), at the cost of ~3.7x more wall-clock time and ~8.4x more tokens for the whole
run. That with-simulation number predates two fixes made after this run (a VCD-based debug context
that was leaking information from the hidden reference module, and a fallback for when the model
diagnoses a fix correctly but fails to apply it) — a rerun with the current code would likely score
higher, but a clean, complete 156-problem run with that code doesn't exist yet.

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
