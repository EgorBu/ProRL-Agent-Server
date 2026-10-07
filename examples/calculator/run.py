#!/usr/bin/env python3
"""Run the calculator demo across every harness and print a comparison table.

Each harness gets a tiny `calculator.py` with parser stubs, edits it, and the
evaluator runs `python3 test_calculator.py`. All harnesses are submitted at
once; live progress and per-session detail are visible in the dashboard
(`polar dashboard -c examples/calculator/topology.vllm.yaml`).

    uv run python examples/calculator/run.py                 # docker (default)
    uv run python examples/calculator/run.py --backend apptainer
    uv run python examples/calculator/run.py --harness codex # Codex-only smoke test
    uv run python examples/calculator/run.py --harness nemo_fabric  # every Fabric adapter
    uv run python examples/calculator/run.py --harness nemo_fabric --fabric-adapter claude
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXAMPLE_DIR / "assets"
TEST_FILE = ASSETS_DIR / "test_calculator.py"
STARTER_FILE = ASSETS_DIR / "calculator.py"
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
RUNTIME_IMAGE = "polar-localhost-calculator:latest"
NUM_SAMPLES = 4
# Generous budget: INIT install (npm / pip / venv) shares the per-task budget
# with the agent run and evaluation.
TIMEOUT_SECONDS = 1200.0
POLL_INTERVAL_SECONDS = 10.0
CODEX_VERSION = "0.125.0"
CODEX_REASONING_EFFORT = "xhigh"

HARNESSES = (
    "claude_code",
    "codex",
    "gemini_cli",
    "opencode",
    "pi",
    "qwen_code",
    "openhands_sdk",
    "openclaw",
    "hermes",
    "mini_swe_agent",
    "nemo_fabric",
)

INSTRUCTION = """\
`calculator.py` has a `Calculator` class with a tokenizer and three stub methods.
Each stub is marked with a `# TODO` comment and returns `0`.

Implement the three methods to build a recursive-descent expression parser:

1. `_parse_expr`  — handle `+` and `-` by calling `_parse_term`
2. `_parse_term`  — handle `*` and `/` (integer division) by calling `_parse_factor`
3. `_parse_factor` — handle integer literals and parenthesized sub-expressions

Also fix `__call__` to return the parsed value instead of `0`.

Requirements:
- Work only in `/polar/session/workspace/calculator.py`.
- Keep the existing file structure, `_tokenize`, `_peek`, and `_consume` as-is.
- Do not add imports.
- Use `//` for division (integer division).
- You must make actual edits. An empty git diff fails the task.

After editing, run `python3 test_calculator.py` to test.
"""

# Per-harness INIT install command. npm CLIs install globally into
# ~/.local/bin; the Python agents install via pip (hermes and mini-swe-agent from
# PyPI into ~/.local, openhands-sdk into ~/.venv where its harness looks for the
# interpreter). The 3.12 runtime image satisfies their Python >=3.11 floor, so a
# plain pip install works here (the tmax example needs uv for its 3.10 images).
# Pinned versions keep the quickstart stable. Bump intentionally.
HARNESS_INSTALL: dict[str, str] = {
    "claude_code": "npm install -g @anthropic-ai/claude-code@2.1.111",
    "codex": f"npm install -g @openai/codex@{CODEX_VERSION}",
    "gemini_cli": "npm install -g @google/gemini-cli@0.38.1",
    "opencode": "npm install -g opencode-ai@1.4.6",
    "pi": "npm install -g @mariozechner/pi-coding-agent@0.67.68",
    "qwen_code": "npm install -g @qwen-code/qwen-code@0.14.5",
    "openclaw": "npm install -g openclaw@2026.5.27",
    "hermes": "python3 -m pip install --user --quiet hermes-agent==0.15.1",
    "mini_swe_agent": "python3 -m pip install --user --quiet mini-swe-agent==2.4.2",
    # Pin sdk + tools to the same version. Unpinned, pip resolves a mismatched
    # pair (sdk 1.17 + tools 1.24) whose imports break; the latest 1.24 needs
    # Python 3.13 (lmnr dep conflict on 3.12), so pin to 1.17.0 for this image.
    "openhands_sdk": (
        "python3 -m venv $HOME/.venv && "
        "$HOME/.venv/bin/pip install --quiet "
        "openhands-sdk==1.17.0 openhands-tools==1.17.0"
    ),
}

# NeMo Fabric adapters, all driven by the single `nemo_fabric` harness: only
# `settings.adapter` changes between them. INIT installs Fabric plus one adapter
# and its harness into ~/.venv, which is first on PATH, so the preset's default
# `python3` is the Fabric interpreter. The PyPI nightly is built from
# FABRIC_COMMIT; TypeScript adapters (Kilo is not on npm) build from it.
FABRIC_VERSION = "0.5.0a20261006"
FABRIC_COMMIT = "8127fbf2b3c20c42eaaa66e0a014b95ccb5af0c8"
_FABRIC_SRC = "$HOME/nemo-fabric"
_HERMES_SRC = "$HOME/hermes-agent"
_NODE24 = "https://nodejs.org/dist/v24.16.0/node-v24.16.0-linux-x64.tar.xz"


def _fabric_install(*requirements: str, then: str = "") -> str:
    pins = " ".join(f"'{requirement}'" for requirement in requirements)
    command = (
        "python3 -m venv $HOME/.venv && $HOME/.venv/bin/pip install --quiet "
        f"--disable-pip-version-check 'nemo-fabric=={FABRIC_VERSION}' {pins}"
    )
    return f"{command} && {then}" if then else command


def _fabric_ts_install(adapter: str, *peers: str) -> str:
    """Build a TypeScript adapter at FABRIC_COMMIT and register its descriptor.

    Fabric scans ``<venv>/share/nemo-fabric`` for descriptors and resolves the
    runner from the real path of a symlinked one, so no discovery config is needed.
    """
    ts = f"{_FABRIC_SRC}/adapters/typescript"
    workspace = f"nemo-fabric-adapters-{adapter}"
    quiet = "--ignore-scripts --no-audit --no-fund --loglevel=error"
    steps = [
        f"git clone -q --filter=blob:none https://github.com/NVIDIA/NeMo-Fabric {_FABRIC_SRC}",
        f"git -C {_FABRIC_SRC} checkout -q {FABRIC_COMMIT}",
        f"npm ci --prefix {_FABRIC_SRC}/adapter-contract/typescript {quiet}",
        f"npm ci --prefix {ts} --workspace {workspace} --include-workspace-root {quiet}",
        f"npm run build --silent --prefix {_FABRIC_SRC}/adapter-contract/typescript",
        f"npm run build --silent --prefix {ts} --workspace nemo-fabric-adapters-common",
        f"npm run build --silent --prefix {ts} --workspace {workspace}",
        *([f"npm install --prefix {ts} --no-save {quiet} {' '.join(peers)}"] if peers else []),
        "mkdir -p $HOME/.venv/share/nemo-fabric",
        f"ln -s {ts}/{adapter}/{adapter}.fabric-adapter.json $HOME/.venv/share/nemo-fabric/",
    ]
    return _fabric_install(then=" && ".join(steps))


FABRIC_INSTALL: dict[str, str] = {
    "claude": _fabric_install(f"nemo-fabric-adapters-claude[harness]=={FABRIC_VERSION}"),
    "codex": _fabric_install(f"nemo-fabric-adapters-codex[harness]=={FABRIC_VERSION}"),
    "deepagents": _fabric_install(
        f"nemo-fabric-adapters-deepagents[harness]=={FABRIC_VERSION}"
    ),
    # Hermes >=0.20 is not on PyPI and refuses wheel builds; install the tag editable.
    "hermes": (
        "git clone -q --depth 1 --branch v2026.9.24 "
        f"https://github.com/NousResearch/hermes-agent {_HERMES_SRC} && "
        + _fabric_install(f"nemo-fabric-adapters-hermes=={FABRIC_VERSION}")
        + f' -e "{_HERMES_SRC}[mcp]"'
    ),
    "mini_swe_agent": _fabric_install(
        f"nemo-fabric-adapters-mini-swe-agent[harness]=={FABRIC_VERSION}"
    ),
    "nooa": _fabric_install(f"nemo-fabric-adapters-nooa[harness]=={FABRIC_VERSION}"),
    "nooa_bench": _fabric_install(f"nemo-fabric-adapters-nooa[harness]=={FABRIC_VERSION}"),
    # The adapter pins openclaw 2026.9.4, which needs Node >=24.16 (image has 22).
    "openclaw": _fabric_install(
        f"nemo-fabric-adapters-openclaw=={FABRIC_VERSION}",
        then=(
            f"curl -fsSL {_NODE24} | tar -xJ -C $HOME/.local --strip-components=1 && "
            "npm install -g openclaw@2026.9.4"
        ),
    ),
    "openhands": _fabric_install(
        f"nemo-fabric-adapters-openhands=={FABRIC_VERSION}",
        "openhands-sdk==1.50.0",
        "openhands-tools==1.50.0",
    ),
    "cline": _fabric_ts_install("cline", "@cline/sdk@0.0.83"),
    "kilo": _fabric_ts_install("kilo", "@kilocode/cli@7.7.12"),
    "opencode": "npm install -g bun@1.4.2 && " + _fabric_ts_install("opencode"),
    "pi": _fabric_ts_install("pi"),
    "qwen": _fabric_ts_install("qwen"),
}

# Model name the harness CLI sends; the gateway rewrites it to the served model.
HARNESS_MODEL: dict[str, str] = {
    "claude_code": "claude-opus-4-5",
    "codex": "openai/gpt-5.5",
    "gemini_cli": "gemini-2.5-flash-lite",
    "opencode": "openai/gpt-5.4",
    "pi": "openai/gpt-5.4",
    "qwen_code": "qwen3-coder-plus",
    "openhands_sdk": "openai/gpt-5.4",
    "openclaw": "openai/gpt-5.4",
    "hermes": "openai/gpt-5.4",
    "mini_swe_agent": "openai/gpt-5.4",
}

# INIT stage: install the harness CLI, then set up a clean git workspace.
_WORKSPACE_PREPARE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
    "cd /polar/session/workspace && "
    "git init -q && "
    "git config user.email 'polar@test' && "
    "git config user.name 'Polar'"
)

# Config/cache dirs that can leak into the workspace git diff.
_EVAL_EXCLUDES: dict[str, list[str]] = {
    "claude_code": [".claude/**", "**/.claude/**"],
    "codex": [".codex/**", "**/.codex/**"],
    "gemini_cli": [".gemini/**", "**/.gemini/**"],
    "opencode": [".opencode/**", "**/.opencode/**", ".config/opencode/**"],
    "pi": [".pi/**", "**/.pi/**"],
    "qwen_code": [".qwen/**", "**/.qwen/**"],
    "openclaw": [".openclaw/**", "**/.openclaw/**"],
    "hermes": [".hermes/**", "**/.hermes/**"],
    "openhands_sdk": [".openhands/**", "**/.openhands/**"],
    "mini_swe_agent": [".mini-swe-agent/**", "**/.mini-swe-agent/**", ".config/mini-swe-agent/**"],
}
_EVAL_EXCLUDES["nemo_fabric"] = sorted({p for patterns in _EVAL_EXCLUDES.values() for p in patterns})
_COMMON_EXCLUDES = [
    "node_modules/**",
    "**/node_modules/**",
    ".cache/**",
    "**/.cache/**",
    ".venv/**",
    "**/.venv/**",
]


def runtime_image_for_backend(backend: str) -> str:
    if backend == "apptainer":
        return f"docker-daemon:{RUNTIME_IMAGE}"
    return RUNTIME_IMAGE


def build_task_payload(run: str, batch_id: str, backend: str) -> dict[str, Any]:
    """Build one task; ``run`` is a harness name or ``nemo_fabric:<adapter>``."""
    harness, _, adapter = run.partition(":")
    if adapter:
        agent: dict[str, Any] = {"harness": harness, "settings": {"adapter": adapter}}
        install = FABRIC_INSTALL[adapter]
    else:
        agent = {"harness": harness, "model_name": HARNESS_MODEL[harness]}
        install = HARNESS_INSTALL[harness]
    if harness == "codex":
        agent["settings"] = {
            "version": CODEX_VERSION,
            "reasoning_effort": CODEX_REASONING_EFFORT,
        }

    return {
        "task_id": f"calculator-{run.replace(':', '-')}-{batch_id}",
        "instruction": INSTRUCTION,
        "num_samples": NUM_SAMPLES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "runtime": {
            "backend": backend,
            "image": runtime_image_for_backend(backend),
            "prepare": [
                {"type": "exec", "command": f"{install} && {_WORKSPACE_PREPARE}"},
                {
                    "type": "upload_file",
                    "source": str(TEST_FILE),
                    "target": "/polar/session/workspace/test_calculator.py",
                },
                {
                    "type": "upload_file",
                    "source": str(STARTER_FILE),
                    "target": "/polar/session/workspace/calculator.py",
                },
                {
                    "type": "exec",
                    "command": "cd /polar/session/workspace && git add -A && git commit -qm 'initial'",
                },
            ],
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": agent,
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "test_on_output",
            "config": {
                "repo_dir": "/polar/session/workspace",
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
                "test_command": "cd /polar/session/workspace && python3 test_calculator.py && echo 'PASSED test_calculator'",
                "test_timeout": 60.0,
                "expected_output_json": {"test_calculator": "PASSED"},
                "exclude_patterns": [*_COMMON_EXCLUDES, *_EVAL_EXCLUDES[harness]],
            },
            "refresh_runtime": True,
        },
    }


def session_reward(session: dict[str, Any]) -> float | None:
    traces = (session.get("trajectory") or {}).get("traces") or []
    reward = traces[-1].get("reward") if traces else None
    return float(reward) if isinstance(reward, (int, float)) else None


def print_comparison(finished: dict[str, dict[str, Any]], elapsed: float) -> None:
    header = f"{'Harness':<28} {'Reward':>8}  {'Done':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for harness, result in finished.items():
        sessions = result.get("results") or []
        rewards = [r for r in (session_reward(s) for s in sessions) if r is not None]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        done = sum(1 for s in sessions if s.get("status") == "COMPLETED")
        print(f"{harness:<28} {mean:>8.3f}  {done:>2}/{len(sessions):<2}")
    print("=" * len(header))
    print(f"Wall time: {elapsed:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("-c", "--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument(
        "--harness",
        action="append",
        choices=HARNESSES,
        help="Run only this harness. Repeat to select more than one.",
    )
    parser.add_argument(
        "--fabric-adapter",
        action="append",
        choices=sorted(FABRIC_INSTALL),
        help="With nemo_fabric, run only this Fabric adapter. Repeatable.",
    )
    args = parser.parse_args()
    backend = args.backend
    fabric_adapters = args.fabric_adapter or sorted(FABRIC_INSTALL)
    selected_harnesses = tuple(
        run
        for harness in args.harness or HARNESSES
        for run in (
            [f"nemo_fabric:{adapter}" for adapter in fabric_adapters]
            if harness == "nemo_fabric"
            else [harness]
        )
    )

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(args.topology).rollout.public_url
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print(f"Submitting {len(selected_harnesses)} harnesses to {rollout_url} (backend={backend})")
    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}
        for harness in selected_harnesses:
            payload = build_task_payload(harness, batch_id, backend)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[harness] = resp.json()["task_id"]
            print(f"  {harness:<28} -> {task_ids[harness]}")

        print(f"\nPolling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        finished: dict[str, dict[str, Any]] = {}
        while len(finished) < len(selected_harnesses):
            time.sleep(POLL_INTERVAL_SECONDS)
            for harness, tid in task_ids.items():
                if harness in finished:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    finished[harness] = status
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {harness} done")
        elapsed = time.monotonic() - t0

    print_comparison(finished, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
