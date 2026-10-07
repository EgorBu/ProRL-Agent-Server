from __future__ import annotations

import json
import os
import runpy
import sys
import types
from pathlib import Path

import pytest

from polar.agent import AgentRunResult, AgentSpec, create_harness
from polar.agent.presets import nemo_fabric
from polar.agent.presets.nemo_fabric import ADAPTERS, NemoFabricHarness
from polar.runtime.models import ExecResult, RuntimeSpec

pytestmark = pytest.mark.unit

WORKSPACE = "/polar/session/workspace"


def harness(adapter: str, **spec) -> NemoFabricHarness:
    settings = {"adapter": adapter, **spec.pop("settings", {})}
    return create_harness(AgentSpec(harness="nemo_fabric", settings=settings, **spec))


def test_factory_and_required_adapter() -> None:
    assert isinstance(harness("hermes"), NemoFabricHarness)
    with pytest.raises(ValueError, match="settings.adapter"):
        create_harness(AgentSpec(harness="nemo_fabric"))
    with pytest.raises(ValueError, match="Unknown nemo_fabric adapter"):
        harness("claude-code")


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_adapter_routes_its_model_to_the_gateway(name: str) -> None:
    adapter = ADAPTERS[name]
    config = harness(name).fabric_config(WORKSPACE)

    if "workflow" in config:  # workflow targets replace the harness selector
        assert "harness" not in config
    else:
        assert config["harness"]["adapter_id"] == adapter.adapter_id
    assert config["environment"]["workspace"] == WORKSPACE
    model = config["models"]["default"]
    assert model["provider"] == adapter.provider
    assert model["model"] == adapter.model
    if adapter.api == "anthropic":
        assert model["base_url"] == "${ANTHROPIC_BASE_URL}"
        assert model["api_key_env"] == "ANTHROPIC_API_KEY"
    else:
        assert model["base_url"] == "${OPENAI_BASE_URL}"
        assert model["api_key_env"] == "OPENAI_API_KEY"
    # Adapter ids and short names resolve to the same config.
    by_id = harness(adapter.adapter_id).fabric_config(WORKSPACE)
    assert {**by_id, "metadata": None} == {**config, "metadata": None}


def test_agent_spec_and_overlay_compose_over_adapter_defaults() -> None:
    h = harness(
        "claude",
        model_name="claude-sonnet-4-5",
        env={"FOO": "bar", "IS_SANDBOX": "0"},
        skills_path="/polar/skills",
        mcp_servers=[
            {"name": "fs", "transport": "stdio", "command": "mcp-fs", "args": ["--root", "/"]},
            {"name": "web", "transport": "streamable-http", "url": "http://127.0.0.1:9000/mcp"},
        ],
        settings={
            "config": {
                "runtime": {"max_turns": 30},
                "harness": {"settings": {"max_budget_usd": 1}},
                "instructions": {"system": {"mode": "append", "content": "Be brief."}},
                "metadata": None,
            }
        },
    )
    config = h.fabric_config(WORKSPACE)

    assert config["models"]["default"]["model"] == "claude-sonnet-4-5"
    assert config["harness"]["settings"] == {
        "permission_mode": "bypassPermissions",
        "max_budget_usd": 1,
    }
    assert config["environment"]["env"] == {
        "IS_SANDBOX": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "FOO": "bar",
    }
    assert config["runtime"] == {"artifacts": nemo_fabric.ARTIFACTS_DIR, "max_turns": 30}
    assert config["instructions"]["system"]["mode"] == "append"
    assert "metadata" not in config
    assert config["skills"] == {"paths": ["/polar/skills"]}
    assert config["mcp"]["servers"] == {
        "fs": {"transport": "stdio", "url": "mcp-fs", "args": ["--root", "/"]},
        "web": {"transport": "streamable-http", "url": "http://127.0.0.1:9000/mcp"},
    }


def test_third_party_adapter_id_gets_openai_routing() -> None:
    config = harness("acme.fabric.agent", model_name="m").fabric_config(WORKSPACE)
    assert config["harness"] == {"adapter_id": "acme.fabric.agent"}
    assert config["models"]["default"]["base_url"] == "${OPENAI_BASE_URL}"


class FakeRuntime:
    def __init__(self, stdout: str | None = None) -> None:
        self.spec = RuntimeSpec(image="img", workdir=WORKSPACE)
        self.runtime_session_dir = "/polar/session"
        self.stdout = stdout
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []

    async def exec(self, command: str, **_: object) -> ExecResult:
        self.commands.append(command)
        return ExecResult(return_code=0, stdout=self.stdout)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        self.uploads.append((local_path, remote_path))


@pytest.mark.asyncio
async def test_setup_writes_config_and_runner_then_runs_it() -> None:
    h = harness("codex", settings={"python": "$HOME/.venv/bin/python"})
    runtime = FakeRuntime()
    await h.setup(runtime)

    (command,) = runtime.commands
    written = json.loads(command.split("<< 'POLARCFG'\n", 1)[1].rsplit("\nPOLARCFG", 1)[0])
    assert written == h.fabric_config(WORKSPACE)
    ((local, remote),) = runtime.uploads
    assert Path(local).name == "nemo_fabric_runner.py" and Path(local).is_file()
    assert remote == f"{nemo_fabric.FABRIC_DIR}/runner.py"

    (step,) = h.run_steps("fix it; don't break $HOME")
    assert step.command.startswith(f"$HOME/.venv/bin/python {nemo_fabric.FABRIC_DIR}/runner.py ")
    assert step.command.endswith("'fix it; don'\"'\"'t break $HOME'")


@pytest.mark.asyncio
async def test_postprocess_surfaces_fabric_error_on_failure() -> None:
    error = {"code": "claude_invalid_configuration", "message": "bad base url"}
    runtime = FakeRuntime(stdout=json.dumps({"status": "failed", "error": error}))
    failed = AgentRunResult(status="failed", return_code=1, error="step 0 exited with code 1")
    await harness("claude").postprocess(runtime, failed)
    assert failed.error == (
        "step 0 exited with code 1: [claude_invalid_configuration] bad base url"
    )

    completed = AgentRunResult(status="completed", return_code=0)
    await harness("claude").postprocess(runtime, completed)
    assert completed.error is None


def run_runner(tmp_path, monkeypatch, fabric_run):
    """Execute the runner script as __main__ against a stub ``nemo_fabric``."""
    seen = {}

    class FakeFabric:
        async def run(self, config, *, base_dir, input):
            seen.update(config=config, base_dir=base_dir, input=input)
            return fabric_run()

    fake = types.ModuleType("nemo_fabric")
    fake.Fabric = FakeFabric
    fake.FabricConfig = types.SimpleNamespace(
        model_validate=lambda raw: types.SimpleNamespace(
            raw=raw, environment=types.SimpleNamespace(**raw["environment"])
        )
    )
    monkeypatch.setitem(sys.modules, "nemo_fabric", fake)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw:8100/v1")
    monkeypatch.setenv("HOME", "/home/polar")
    monkeypatch.setenv("ADAPTER_PYTHON", "unset")
    monkeypatch.delenv("ADAPTER_PYTHON")  # restored (deleted) after the runner sets it

    config = harness("deepagents").fabric_config(WORKSPACE)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result_path = tmp_path / "out" / "result.json"
    monkeypatch.setattr(
        sys, "argv",
        ["runner", "--config", str(config_path), "--result", str(result_path), "do it"],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(Path(nemo_fabric.__file__).with_name("nemo_fabric_runner.py")),
                       run_name="__main__")
    return exit_info.value.code, seen, json.loads(result_path.read_text())


def test_runner_expands_gateway_env_and_reports_status(tmp_path, monkeypatch, capsys) -> None:
    class FakeResult:
        def to_mapping(self):
            return {"status": "succeeded", "output": {"response": "done"}}

    code, seen, result = run_runner(tmp_path, monkeypatch, FakeResult)

    assert code == 0
    raw = seen["config"].raw
    assert raw["models"]["default"]["base_url"] == "http://gw:8100/v1"
    assert raw["environment"]["env"]["HOME"] == "/home/polar"
    assert seen["base_dir"] == WORKSPACE and seen["input"] == "do it"
    assert result["status"] == "succeeded"
    assert json.loads(capsys.readouterr().out)["response"] == "done"
    assert os.environ["ADAPTER_PYTHON"] == sys.executable


def test_runner_records_exceptions_as_failed_results(tmp_path, monkeypatch) -> None:
    def boom():
        raise RuntimeError("adapter nvidia.fabric.pi not found")

    code, _, result = run_runner(tmp_path, monkeypatch, boom)

    assert code == 1
    assert result == {
        "status": "failed",
        "error": {
            "stage": "runner",
            "code": "RuntimeError",
            "message": "adapter nvidia.fabric.pi not found",
        },
    }
