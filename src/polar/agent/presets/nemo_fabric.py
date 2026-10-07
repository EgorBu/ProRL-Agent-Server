"""NeMo Fabric harness — https://github.com/NVIDIA/NeMo-Fabric

One preset for every harness Fabric integrates. ``settings.adapter`` selects
the Fabric adapter (``claude``, ``codex``, ``hermes``, ``openhands``, ...);
Polar turns the agent spec into a ``FabricConfig``, routes the adapter's model
to the gateway endpoint matching its wire protocol, and runs
``Fabric().run()`` inside the runtime through ``nemo_fabric_runner.py``.

``settings.config`` is deep-merged over the generated config (``None`` deletes
a key), so every ``FabricConfig`` field stays reachable. ``settings.python``
names the runtime interpreter that has ``nemo-fabric`` and the adapter
installed (default ``python3``).
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from polar.agent.base import BaseHarness
from polar.agent.models import AgentRunResult, AgentSpec, MCPServerSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput

FABRIC_DIR = f"{RUNTIME_SESSION_DIR}/.fabric"
ARTIFACTS_DIR = f"{RUNTIME_AGENT_LOG_DIR}/fabric"
_RUNNER = Path(__file__).with_name("nemo_fabric_runner.py")

# Gateway endpoint per wire protocol. The runner expands ${VAR} at exec time;
# OPENAI_BASE_URL already ends in /v1 (chat completions and responses).
_GATEWAY: dict[str, dict[str, str]] = {
    "openai": {"base_url": "${OPENAI_BASE_URL}", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "${ANTHROPIC_BASE_URL}", "api_key_env": "ANTHROPIC_API_KEY"},
}


@dataclass(frozen=True)
class FabricAdapter:
    """How Polar drives one Fabric adapter unattended through the gateway.

    ``defaults`` is a ``FabricConfig`` fragment merged under the user's
    ``settings.config`` (permission bypass, env the harness needs, ...).
    """

    adapter_id: str
    api: Literal["openai", "anthropic"] = "openai"
    provider: str = "openai"
    model: str = "gpt-5.4"
    defaults: dict[str, Any] = field(default_factory=dict)


ADAPTERS: dict[str, FabricAdapter] = {
    # Custom provider names make Claude/Codex use base_url + api_key_env
    # instead of their native login flows.
    "claude": FabricAdapter(
        "nvidia.fabric.claude",
        api="anthropic",
        provider="polar",
        model="claude-opus-4-5",
        defaults={
            "harness": {"settings": {"permission_mode": "bypassPermissions"}},
            # Claude's child env is an allowlist plus environment.env.
            "environment": {
                "env": {"IS_SANDBOX": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
            },
        },
    ),
    "codex": FabricAdapter(
        "nvidia.fabric.codex",
        provider="polar",
        model="gpt-5.5",
        # Codex's landlock/seccomp sandbox does not work in unprivileged containers.
        defaults={
            "harness": {"settings": {"sandbox": "danger-full-access", "approval_mode": "deny_all"}}
        },
    ),
    "deepagents": FabricAdapter(
        "nvidia.fabric.langchain.deepagents",
        defaults={
            # The default backend is a virtual filesystem without a shell; the
            # shell backend sees only environment.env and needs an explicit tool list.
            "harness": {"settings": {"deepagents": {"backend": {"type": "local_shell"}}}},
            "environment": {"env": {"PATH": "${PATH}", "HOME": "${HOME}"}},
            "tools": {
                "enabled": [
                    "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
                    "execute", "write_todos", "task",
                ]
            },
        },
    ),
    "hermes": FabricAdapter(
        "nvidia.fabric.hermes",
        provider="custom",
        # Hermes caps completions at 512 tokens unless told otherwise.
        defaults={"models": {"default": {"max_tokens": 16384}}},
    ),
    "mini_swe_agent": FabricAdapter(
        "nvidia.fabric.mini-swe-agent",
        defaults={
            "environment": {
                "env": {"MSWEA_COST_TRACKING": "ignore_errors", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
            }
        },
    ),
    # NOOA's CodeAct coding agent is a workflow target of the shared adapter.
    "nooa": FabricAdapter(
        "nvidia.fabric.nooa",
        defaults={"harness": None, "workflow": {"target_id": "nvidia.nooa.coding-agent"}},
    ),
    "nooa_bench": FabricAdapter("nvidia.fabric.nooa.bench-agent"),
    "openclaw": FabricAdapter("nvidia.fabric.openclaw", provider="polar"),
    "openhands": FabricAdapter(
        "nvidia.fabric.openhands",
        defaults={"environment": {"env": {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}}},
    ),
    # TypeScript adapters (node/bun processes).
    "cline": FabricAdapter("nvidia.fabric.cline", provider="openai-compatible"),
    "kilo": FabricAdapter(
        "nvidia.fabric.kilo",
        # "openai" would collide with Kilo's built-in OpenAI provider.
        provider="polar",
        defaults={
            # The gateway is plain HTTP on loopback.
            "harness": {"settings": {"allow_insecure_http_model_endpoint": True}},
            # Unlisted tools fall back to Kilo's interactive "ask", which never
            # resolves headless; listing them makes them "allow".
            "tools": {
                "enabled": [
                    # Kilo offers edit/multiedit or apply_patch depending on the model.
                    "bash", "read", "write", "edit", "multiedit", "apply_patch",
                    "grep", "glob", "list", "todowrite", "task", "skill",
                ]
            },
        },
    ),
    "opencode": FabricAdapter("nvidia.fabric.opencode"),
    # Pi only accepts models from its catalog; base_url overrides the endpoint.
    "pi": FabricAdapter("nvidia.fabric.pi"),
    "qwen": FabricAdapter(
        "nvidia.fabric.qwen",
        # Qwen Code's default mode denies tool calls that need approval.
        defaults={"harness": {"settings": {"permission_mode": "yolo"}}},
    ),
}


class NemoFabricHarness(BaseHarness):
    """Run any NeMo Fabric adapter through ``Fabric().run()`` in the runtime."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        name = self.settings.get("adapter")
        if not name:
            raise ValueError("nemo_fabric harness requires settings.adapter")
        self.adapter_name = str(name)
        self.adapter = resolve_adapter(self.adapter_name)
        self.python = str(self.settings.get("python", "python3"))

    async def setup(self, runtime: BaseRuntime) -> None:
        workspace = runtime.spec.workdir or runtime.runtime_session_dir
        config = json.dumps(self.fabric_config(workspace))
        await runtime.exec(
            f"mkdir -p {FABRIC_DIR} {ARTIFACTS_DIR} && "
            f"cat > {FABRIC_DIR}/config.json << 'POLARCFG'\n{config}\nPOLARCFG"
        )
        await runtime.upload_file(str(_RUNNER), f"{FABRIC_DIR}/runner.py")

    def run_steps(self, instruction: str) -> list[ExecInput]:
        return [
            ExecInput(
                command=(
                    f"{self.python} {FABRIC_DIR}/runner.py "
                    f"--config {FABRIC_DIR}/config.json "
                    f"--result {ARTIFACTS_DIR}/result.json "
                    f"{shlex.quote(instruction)}"
                ),
            )
        ]

    async def postprocess(self, runtime: BaseRuntime, result: AgentRunResult) -> None:
        # Polar persists only AgentRunResult.error; carry Fabric's normalized error.
        if result.status != "failed":
            return
        out = await runtime.exec(f"cat {ARTIFACTS_DIR}/result.json")
        if out.return_code == 0 and out.stdout:
            error = json.loads(out.stdout).get("error")
            if error:
                result.error = f"{result.error}: [{error.get('code')}] {error.get('message')}"

    def fabric_config(self, workspace: str) -> dict[str, Any]:
        adapter = self.adapter
        config: dict[str, Any] = {
            "metadata": {"name": f"polar-{self.adapter_name.rsplit('.', 1)[-1]}"},
            "harness": {"adapter_id": adapter.adapter_id},
            "runtime": {"artifacts": ARTIFACTS_DIR},
            "environment": {"provider": "local", "workspace": workspace, "artifacts": ARTIFACTS_DIR},
            "models": {
                "default": {
                    "provider": adapter.provider,
                    "model": self.model_name or adapter.model,
                    **_GATEWAY[adapter.api],
                }
            },
        }
        config = deep_merge(config, adapter.defaults)
        if self.env:
            config = deep_merge(config, {"environment": {"env": self.env}})
        if self.skills_path:
            config["skills"] = {"paths": [self.skills_path]}
        if self.mcp_servers:
            config["mcp"] = {
                "servers": {server.name: _mcp_server(server) for server in self.mcp_servers}
            }
        return deep_merge(config, self.settings.get("config") or {})


def resolve_adapter(name: str) -> FabricAdapter:
    """Look up an adapter by short name or Fabric adapter id.

    Unlisted ids (third-party adapters) get OpenAI-compatible gateway routing;
    override anything else through ``settings.config``.
    """
    if name in ADAPTERS:
        return ADAPTERS[name]
    for adapter in ADAPTERS.values():
        if adapter.adapter_id == name:
            return adapter
    if "." not in name:
        raise ValueError(
            f"Unknown nemo_fabric adapter {name!r}; use one of {sorted(ADAPTERS)} "
            "or a full Fabric adapter id"
        )
    return FabricAdapter(name)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mcp_server(server: MCPServerSpec) -> dict[str, Any]:
    # Fabric carries the stdio executable in ``url``.
    if server.transport == "stdio":
        return {"transport": "stdio", "url": server.command, "args": server.args}
    return {"transport": server.transport, "url": server.url}
