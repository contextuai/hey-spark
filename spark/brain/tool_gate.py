"""Compatibility shim for the Phase 2 deny-by-default gate.

The ``PreToolUse`` hook now lives in :mod:`spark.brain.confirm` (P3.1). ``make_tool_gate`` returns
a :class:`~spark.brain.confirm.ConfirmationGate` with no confirmer. It passes read-only tools and
denies everything else without asking, which is exactly the Phase 2 behaviour. New code should
build ``ConfirmationGate(policy, confirmer)`` directly.
"""

from __future__ import annotations

from spark.brain.confirm import DENY_REASONS, ConfirmationGate, HookCallback, deny
from spark.brain.mcp_config import ToolPolicy

__all__ = ["DENY_REASONS", "HookCallback", "deny", "make_tool_gate"]


def make_tool_gate(policy: ToolPolicy) -> HookCallback:
    """Read-only tools pass (``{}``); side-effecting, destructive and unknown tools are denied."""
    return ConfirmationGate(policy, None)
