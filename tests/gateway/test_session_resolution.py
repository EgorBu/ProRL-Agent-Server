from __future__ import annotations

import pytest

from polar.gateway.session import SessionRegistry, resolve_session_id

pytestmark = pytest.mark.unit

POLAR_SESSION = "sk-polar-1234"


@pytest.fixture
def registry() -> SessionRegistry:
    registry = SessionRegistry()
    registry.register(POLAR_SESSION)
    return registry


def test_agent_session_header_does_not_hijack_the_api_key_session(registry) -> None:
    # OpenCode 2.x stamps its own session id on every request.
    headers = {"Authorization": f"Bearer {POLAR_SESSION}", "x-session-id": "ses_opencode"}
    assert resolve_session_id(registry, headers, {}) == POLAR_SESSION
    assert registry.get("ses_opencode") is None


def test_registered_explicit_session_wins_over_api_key(registry) -> None:
    registry.register("sk-polar-other")
    headers = {"Authorization": f"Bearer {POLAR_SESSION}", "x-session-id": "sk-polar-other"}
    assert resolve_session_id(registry, headers, {}) == "sk-polar-other"


def test_unknown_explicit_session_is_opened_without_a_known_api_key(registry) -> None:
    assert resolve_session_id(registry, {"x-session-id": "client-run-7"}, {}) == "client-run-7"
    assert registry.get("client-run-7") is not None
    assert resolve_session_id(registry, {}, {}, query_session_id="q-1") == "q-1"


def test_unknown_credentials_open_a_fresh_session(registry) -> None:
    session_id = resolve_session_id(registry, {"Authorization": "Bearer nope"}, {})
    assert session_id not in (POLAR_SESSION, "nope")
    assert registry.get(session_id) is not None
