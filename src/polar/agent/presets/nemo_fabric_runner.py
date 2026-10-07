"""Run one NeMo Fabric invocation inside a Polar runtime.

The ``nemo_fabric`` preset uploads this file into the runtime and executes it
with the interpreter that has ``nemo-fabric`` installed. It must not import
Polar. ``${VAR}`` references in ``models.*.base_url`` and ``environment.env``
are expanded here because the gateway endpoints only exist in the exec env.
The result file is always written, so the preset can surface Fabric's error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from nemo_fabric import Fabric, FabricConfig


def run(config_path: Path, instruction: str) -> dict[str, Any]:
    raw = json.loads(config_path.read_text())
    for model in raw.get("models", {}).values():
        if model.get("base_url"):
            model["base_url"] = os.path.expandvars(model["base_url"])
    env = (raw.get("environment") or {}).get("env", {})
    env.update({key: os.path.expandvars(value) for key, value in env.items()})
    config = FabricConfig.model_validate(raw)

    # Python adapters run in a host interpreter Fabric resolves from
    # ADAPTER_PYTHON; default it to this one, which has nemo-fabric installed.
    os.environ.setdefault("ADAPTER_PYTHON", sys.executable)
    workspace = config.environment.workspace if config.environment else None
    result = asyncio.run(Fabric().run(config, base_dir=workspace, input=instruction))
    return result.to_mapping()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("instruction")
    args = parser.parse_args()

    try:
        result = run(args.config, args.instruction)
    except Exception as exc:
        traceback.print_exc()
        error = {"stage": "runner", "code": type(exc).__name__, "message": str(exc)}
        result = {"status": "failed", "error": error}

    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2))
    output = result.get("output")
    print(json.dumps({
        "status": result["status"],
        "adapter_id": result.get("adapter_id"),
        "response": output.get("response") if isinstance(output, dict) else output,
        "error": result.get("error"),
        "usage": result.get("usage"),
    }, indent=2))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
