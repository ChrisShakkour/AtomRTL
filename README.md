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
