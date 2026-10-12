"""Run the agent on verilog-eval spec-to-rtl problems and score each one against the hidden testbench.

    python scripts/run_verilog_eval.py [--problems 'Prob0[0-2]*' ...] [--limit N] [--jobs N] [--sim]
                                       [--config cfg.yaml] [--provider ollama|anthropic] [--model NAME]
                                       [--resume-dir runs/<ts>]

Creates runs/<timestamp>/ (see README) with one directory per problem. Each problem runs as its own
`python -m atomrtl.cli` subprocess (hard timeout, isolation) against one shared Ollama server, or against
the Anthropic API with --provider anthropic (needs ANTHROPIC_API_KEY).
By default the agent never sees the testbench or reference: scoring happens only after it finishes.
With --sim (or `simulation.enabled` in the config) the agent itself simulates against the same testbench
and reference and fixes mismatches (it sees only the mismatch report and a waveform table, never their
source), so the pass rate is oracle-assisted and not comparable to a run without it.
"""

import argparse
import fnmatch
import json
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from atomrtl.cli import new_run_dir, write_run_info
from atomrtl.config import PROVIDERS, REPO_ROOT, load_config
from atomrtl.eda import simulate
from atomrtl.llm import ServerKeeper, llm_backend

DEFAULT_DATASET = REPO_ROOT.parent / "verilog-eval" / "dataset_spec-to-rtl"


def find_problems(dataset: Path, patterns: list[str] | None, limit: int | None) -> list[str]:
    names = sorted(p.name.removesuffix("_prompt.txt") for p in dataset.glob("*_prompt.txt"))
    if patterns:
        names = [n for n in names if any(fnmatch.fnmatch(n, pat) or n == pat for pat in patterns)]
    return names[:limit] if limit else names


def run_agent_subprocess(problem: str, dataset: Path, workdir: Path, config_path: Path, timeout: float,
                         sim: bool = False) -> dict:
    cmd = [sys.executable, "-m", "atomrtl.cli", "--spec", str(dataset / f"{problem}_prompt.txt"),
           "--top", "TopModule", "--config", str(config_path), "--workdir", str(workdir), "--no-server"]
    if sim:
        cmd += ["--sim-tb", str(dataset / f"{problem}_test.sv"), "--sim-ref", str(dataset / f"{problem}_ref.sv")]
    start = time.monotonic()
    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir / "agent.log", "w") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        try:
            p = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            # cli exit codes: 0 = pass, 1 = finished without a passing design, 2 = crashed
            agent_status = "completed" if p.returncode in (0, 1) else "error"
        except subprocess.TimeoutExpired:
            agent_status = "timeout"
    return {"agent_status": agent_status, "agent_duration_s": round(time.monotonic() - start, 1)}


def score(problem: str, dataset: Path, workdir: Path, timeout: float = 60) -> dict:
    """Compile the generated code with the hidden testbench + reference and simulate (after the agent is done)."""
    sources = sorted(workdir.glob("*.sv"))
    if not sources:
        return {"status": "no_generated_code"}
    sim = simulate(dataset / f"{problem}_test.sv", dataset / f"{problem}_ref.sv",
                   [Path(s.name) for s in sources], workdir, timeout)
    result = {"status": sim.status}
    if sim.status in ("pass", "fail"):
        result |= {"mismatches": sim.mismatches, "samples": sim.samples}
    (workdir / "validation.log").write_text("\n\n".join(sim.log))
    (workdir / "validation.json").write_text(json.dumps(result, indent=2))
    return result


def run_problem(problem: str, dataset: Path, run_dir: Path, config_path: Path, timeout: float,
                keeper: ServerKeeper, sim: bool = False, retries: int = 1) -> dict:
    """Agent then scoring. A crashed agent (e.g. the LLM server went away) is retried from scratch."""
    workdir = run_dir / problem
    crashes = []
    for _ in range(retries + 1):
        if workdir.exists():
            shutil.rmtree(workdir)  # no leftovers from a crashed or interrupted attempt
        keeper.ensure()  # restart the shared server if it went away
        info = {"problem": problem, **run_agent_subprocess(problem, dataset, workdir, config_path, timeout, sim)}
        if info["agent_status"] != "error":
            break
        crashes.append((workdir / "agent.log").read_text().strip().splitlines()[-1])
    if crashes:
        info["crashes"] = crashes

    result_file = workdir / "result.json"
    if result_file.exists():
        r = json.loads(result_file.read_text())
        info |= {"agent_result": r["status"], "attempts": r["attempts"], "tokens": r.get("tokens", {})}
        if "simulation" in r:
            hist = r["simulation"]["history"]
            info |= {"sim_fixes": r["simulation"]["sim_fixes"], "first_sim": hist[0]["status"] if hist else None}
    info |= score(problem, dataset, workdir)
    (workdir / "status.json").write_text(json.dumps(info, indent=2))
    return info


def is_done(workdir: Path) -> bool:
    """Scored, and the agent didn't crash (crashed problems are rerun on --resume-dir)."""
    status = workdir / "status.json"
    return status.exists() and ((workdir / "result.json").exists() or json.loads(status.read_text())["agent_status"] == "timeout")


def summarize(run_dir: Path, rows: list[dict], sim: bool = False) -> dict:
    by_status = Counter(r["status"] for r in rows)
    tokens = Counter()
    for r in rows:
        tokens.update(r.get("tokens", {}))
    summary = {
        "total": len(rows),
        "passed": by_status["pass"],
        "pass_rate": round(by_status["pass"] / len(rows), 3) if rows else 0,
        "by_status": dict(by_status),
        "agent_self_check_pass": sum(r.get("agent_result") == "pass" for r in rows),
        "avg_attempts": round(sum(r.get("attempts", 0) for r in rows) / len(rows), 2) if rows else 0,
        **({"simulation_enabled": True,
            "avg_sim_fixes": round(sum(r.get("sim_fixes", 0) for r in rows) / len(rows), 2) if rows else 0,
            # First agent simulation failed, final score passes: what the sim-debug loop gained.
            "fixed_by_simulation": sum(r.get("first_sim") not in (None, "pass") and r["status"] == "pass" for r in rows),
            } if sim else {}),
        "avg_agent_duration_s": round(sum(r["agent_duration_s"] for r in rows) / len(rows), 1) if rows else 0,
        "tokens": dict(tokens),
        "problems": sorted(rows, key=lambda r: r["problem"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--problems", nargs="+", help="problem names or globs (default: all)")
    ap.add_argument("--limit", type=int, help="run at most N problems")
    ap.add_argument("--config", type=Path, help="YAML config (see atomrtl/config.py)")
    ap.add_argument("--provider", choices=PROVIDERS, help="LLM backend (default: from the config, else ollama)")
    ap.add_argument("--model", help="override the active provider's model")
    ap.add_argument("--jobs", type=int, help="problems run in parallel (default: the provider's num_parallel)")
    ap.add_argument("--sim", action="store_true",
                    help="let the agent simulate against the problem's testbench and fix mismatches (oracle feedback)")
    ap.add_argument("--timeout", type=float, default=900, help="per-problem agent timeout, seconds (default 900)")
    ap.add_argument("--resume-dir", type=Path,
                    help="finish an interrupted run: skips problems already scored, reruns crashed ones")
    args = ap.parse_args()

    if not args.dataset_dir.is_dir():
        sys.exit(f"dataset not found: {args.dataset_dir} (clone verilog-eval next to AtomRTL, or pass --dataset-dir)")

    if args.resume_dir:
        run_dir = args.resume_dir
        cfg = load_config(run_dir / "config.yaml")
    else:
        cfg = load_config(args.config, provider=args.provider)
        if args.model:
            cfg = cfg.with_model(args.model)
        if args.sim:
            cfg = replace(cfg, simulation=replace(cfg.simulation, enabled=True))
        run_dir = new_run_dir(cfg.runs_dir)
        write_run_info(run_dir, cfg)
    config_path = run_dir / "config.yaml"  # every problem uses exactly this resolved config
    sim = cfg.simulation.enabled

    problems = find_problems(args.dataset_dir, args.problems, args.limit)
    todo = [p for p in problems if not is_done(run_dir / p)]
    jobs = args.jobs or cfg.num_parallel
    print(f"run dir: {run_dir}\nprovider: {cfg.provider}  model: {cfg.model}  simulation: {'on' if sim else 'off'}  "
          f"problems: {len(todo)} to run ({len(problems) - len(todo)} already done)  jobs: {jobs}\n", flush=True)

    with llm_backend(cfg) as keeper:
        with ThreadPoolExecutor(jobs) as pool:
            futures = {pool.submit(run_problem, p, args.dataset_dir, run_dir, config_path, args.timeout, keeper, sim): p
                       for p in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                extra = f"{r['mismatches']}/{r['samples']} mismatches" if "mismatches" in r else ""
                extra += f"  sim_fixes={r['sim_fixes']}" if "sim_fixes" in r else ""
                print(f"[{i:3d}/{len(todo)}] {r['problem']:<32} {r['status']:<18} agent={r.get('agent_result', r['agent_status']):<8} "
                      f"attempts={r.get('attempts', '-')}  {r['agent_duration_s']:6.1f}s  {extra}", flush=True)

    rows = [json.loads(f.read_text()) for p in problems if (f := run_dir / p / "status.json").exists()]
    s = summarize(run_dir, rows, sim)
    print(f"\npass rate: {s['passed']}/{s['total']} = {s['pass_rate']:.1%}   by status: {s['by_status']}")
    print(f"agent self-check pass (compile+lint+interface): {s['agent_self_check_pass']}/{s['total']}   "
          f"avg attempts: {s['avg_attempts']}")
    if sim:
        print(f"simulation: avg sim fixes {s['avg_sim_fixes']}, fixed by simulation: {s['fixed_by_simulation']}")
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
