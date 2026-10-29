"""MCP tools as OpenAI function tools (P5.7e): names, schemas and results.

* **Names.** OpenAI function names must match ``^[a-zA-Z0-9_-]{1,64}$``. Spark's tool names are
  ``mcp__<server>__<tool>``, which usually fit and are used unchanged, so logs, the activity log,
  the policy and the gate all keep speaking the same names. A name with other characters is
  cleaned, and one longer than 64 is cut and given a stable hash suffix. :class:`ToolNameMap`
  maps both ways; collisions get a hash too.
* **Schemas.** An MCP tool's ``inputSchema`` is JSON Schema, which ``parameters`` accepts. Only
  what trips some providers is adjusted: the root must be ``{"type": "object", "properties":
  {...}}``, and ``$schema`` / ``$id`` / ``$comment`` keys are dropped everywhere.
* **Results.** An MCP ``CallToolResult`` (text blocks, images, structured content, ``isError``)
  becomes the text of a ``role: tool`` message, cut to ``max_tool_result_chars`` (the activity
  log keeps the whole, redacted, result).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "FUNCTION_NAME_RE",
    "ToolNameMap",
    "clip_result",
    "convert_schema",
    "function_tool",
    "result_text",
]

FUNCTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_BAD_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_DROP_KEYS = frozenset({"$schema", "$id", "$comment"})
MAX_DESCRIPTION = 1000


def _hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


class ToolNameMap:
    """Reversible tool name <-> function name map. Build it with every offered tool name."""

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._to_fn: dict[str, str] = {}
        self._to_tool: dict[str, str] = {}
        for name in names:
            self.add(name)

    def add(self, name: str) -> str:
        if name in self._to_fn:
            return self._to_fn[name]
        fn = name if FUNCTION_NAME_RE.match(name) else _BAD_CHARS.sub("_", name)
        if len(fn) > 64:
            fn = f"{fn[:55]}_{_hash(name)}"
        if not fn:
            fn = f"tool_{_hash(name)}"
        if fn in self._to_tool and self._to_tool[fn] != name:
            fn = f"{fn[:55]}_{_hash(name)}"
        self._to_fn[name] = fn
        self._to_tool[fn] = name
        return fn

    def function(self, name: str) -> str:
        return self._to_fn.get(name) or self.add(name)

    def tool(self, function_name: str) -> str | None:
        """The tool behind a function name the model called (None: never offered)."""
        return self._to_tool.get(function_name)

    def __contains__(self, function_name: object) -> bool:
        return function_name in self._to_tool

    def __len__(self) -> int:
        return len(self._to_fn)


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _clean(v) for k, v in value.items() if k not in _DROP_KEYS}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def convert_schema(schema: Any) -> dict[str, Any]:
    """MCP ``inputSchema`` -> OpenAI ``parameters``."""
    if not isinstance(schema, Mapping):
        return {"type": "object", "properties": {}}
    out = _clean(schema)
    if out.get("type") != "object":
        # A bare non-object schema can't be function parameters; wrap nothing, accept anything.
        out = {"type": "object", "properties": out.get("properties") or {}}
    if not isinstance(out.get("properties"), Mapping):
        out["properties"] = {}
    required = out.get("required")
    if required is not None and not (
        isinstance(required, list) and all(isinstance(r, str) for r in required)
    ):
        out.pop("required")
    return out


def function_tool(function_name: str, description: str | None, schema: Any) -> dict[str, Any]:
    """One entry of the request's ``tools`` list."""
    text = " ".join((description or "").split())
    if len(text) > MAX_DESCRIPTION:
        text = text[: MAX_DESCRIPTION - 1].rstrip() + "…"
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": text,
            "parameters": convert_schema(schema),
        },
    }


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def result_text(content: Any, structured: Any = None) -> str:
    """The text of MCP result content (text blocks; other blocks as ``[image]`` etc.)."""
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, Mapping):
            kind, text = block.get("type"), block.get("text")
        else:
            kind, text = getattr(block, "type", None), getattr(block, "text", None)
        if kind == "text" and isinstance(text, str):
            parts.append(text)
        elif kind == "resource":
            resource = _field(block, "resource")
            inner = _field(resource, "text")
            parts.append(inner if isinstance(inner, str) else "[resource]")
        else:
            parts.append(f"[{kind or 'content'}]")
    if not any(p.strip() for p in parts) and structured is not None:
        try:
            return json.dumps(structured, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(structured)
    return "\n".join(parts)


def clip_result(text: str, limit: int) -> str:
    """Cut a tool result for the model, saying how much was left out."""
    if len(text) <= limit:
        return text
    kept = text[:limit]
    return f"{kept}\n[... {len(text) - limit} more characters cut; ask a narrower question]"
