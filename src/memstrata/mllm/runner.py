"""MLLM role runner — the single transport that *executes* a role declared in
``memstrata.mllm.roles``.

``roles.py`` is spec-only (which role, which model, which sampling/schema).
This module is the thin, dependency-light execution layer that turns a
``RoleSpec`` + call-site inputs into one OpenAI-compatible chat request and a
validated result. It centralizes three things every hot-path role needs:

    1. **Model routing** — text roles -> the text endpoint/model, VISION roles
       -> the VL endpoint/model (crops/frames become image content parts).
    2. **Sampling contract** — temperature / top_p / max_tokens / thinking /
       response_format are taken from ``RoleSpec.sampling`` (not re-specified at
       every call site), so the registry stays the single source of truth.
    3. **Structured decoding** — when the role emits JSON, the request pins
       ``response_format=json_schema`` and the reply is parsed + shape-checked
       against the caller-supplied schema (or a permissive one derived from
       ``RoleSpec.schema_fields``).

No heavy deps (urllib + base64 + json only), mirroring ``MllmPlanner._call_api``.
Transport is injectable so roles are unit-testable without a live server.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from memstrata.mllm.roles import DEFAULT_MODEL, ROLE_REGISTRY, Modality, RoleSpec, Sampling

# Endpoint / model env keys. Qwen3.5-9B is multimodal, so by default ONE server
# plays every role (text + vision); split endpoints remain possible via env/args.
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"        # unified Qwen3.5-9B


class Transport(Protocol):
    """Minimal chat transport. Returns the assistant message *content* string."""

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        sampling: Sampling,
        schema: dict[str, Any] | None,
        timeout: float,
    ) -> str: ...


@dataclass(slots=True)
class HttpTransport:
    """OpenAI-compatible ``/chat/completions`` over urllib (no SDK dependency)."""

    base_url: str

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        sampling: Sampling,
        schema: dict[str, Any] | None,
        timeout: float,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.response_format == "json_schema" and schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "role_response", "schema": schema, "strict": True},
            }
        # Qwen reasoning toggle (vLLM/SGLang chat_template_kwargs).
        payload["chat_template_kwargs"] = {"enable_thinking": bool(sampling.thinking)}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        return res["choices"][0]["message"]["content"]


def image_content_part(image: str | Path) -> dict[str, Any]:
    """Build an OpenAI vision content part from a local file path (base64 data URL)."""
    p = Path(image)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _permissive_schema(fields: tuple[str, ...]) -> dict[str, Any] | None:
    """Derive a loose object schema from RoleSpec.schema_fields (all required)."""
    if not fields:
        return None
    return {
        "type": "object",
        "properties": {f: {} for f in fields},
        "required": list(fields),
        "additionalProperties": True,
    }


class MllmRoleRunner:
    """Execute registry roles. One runner can serve every role; it picks the
    endpoint/model/sampling from the ``RoleSpec`` at call time.

    Parameters
    ----------
    text_transport / vision_transport:
        Injectable transports. Defaults build ``HttpTransport`` from env
        (``MEMSTRATA_CONTEXT_JUDGER_BASE_URL`` / ``MEMSTRATA_VLM_BASE_URL``).
        Pass a mock in tests.
    text_model / vision_model:
        Override the model name sent to the server (else ``RoleSpec.model`` for
        text; ``vision_model`` for VISION roles).
    """

    def __init__(
        self,
        *,
        text_transport: Transport | None = None,
        vision_transport: Transport | None = None,
        text_model: str | None = None,
        vision_model: str | None = None,
        registry: dict[str, RoleSpec] | None = None,
        timeout: float = 90.0,
    ) -> None:
        text_url = os.environ.get("MEMSTRATA_CONTEXT_JUDGER_BASE_URL") or DEFAULT_BASE_URL
        # Vision defaults to the SAME unified endpoint/model (Qwen3.5-9B is multimodal);
        # override MEMSTRATA_VLM_BASE_URL / MEMSTRATA_VLM_MODEL only to split them out.
        vlm_url = os.environ.get("MEMSTRATA_VLM_BASE_URL") or text_url
        self.text_transport = text_transport or HttpTransport(text_url)
        self.vision_transport = vision_transport or HttpTransport(vlm_url)
        self.text_model = text_model
        self.vision_model = (
            vision_model or os.environ.get("MEMSTRATA_VLM_MODEL") or text_model or DEFAULT_MODEL
        )
        self.registry = registry or ROLE_REGISTRY
        self.timeout = timeout
        self.calls: list[dict[str, Any]] = []  # lightweight per-call ledger

    def _model_for(self, role: RoleSpec) -> str:
        if role.modality == Modality.VISION:
            return self.vision_model
        return self.text_model or role.model

    def _transport_for(self, role: RoleSpec) -> Transport:
        return self.vision_transport if role.modality == Modality.VISION else self.text_transport

    def run(
        self,
        role_key: str,
        *,
        instruction: str,
        images: list[str | Path] | None = None,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        """Run one role. Returns a parsed dict for JSON roles, else the raw text.

        ``instruction`` is the fully-formatted user prompt (call sites own the
        template). ``images`` are local paths, attached only for VISION roles.
        ``schema`` overrides the permissive schema built from schema_fields.
        """
        role = self.registry[role_key]
        content: list[dict[str, Any]] | str
        if role.modality == Modality.VISION and images:
            content = [{"type": "text", "text": instruction}]
            content += [image_content_part(im) for im in images]
        else:
            content = instruction

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        eff_schema = schema
        if role.sampling.response_format == "json_schema" and eff_schema is None:
            eff_schema = _permissive_schema(role.schema_fields)

        raw = self._transport_for(role).chat(
            model=self._model_for(role),
            messages=messages,
            sampling=role.sampling,
            schema=eff_schema,
            timeout=self.timeout,
        )
        self.calls.append({"role": role.id, "key": role_key, "model": self._model_for(role)})

        if role.sampling.response_format == "json_schema":
            result = self._parse_json(role, raw, messages, eff_schema)
            missing = [f for f in role.schema_fields if f not in result]
            if missing:
                raise ValueError(f"role {role.id} reply missing fields {missing}: {result}")
            return result
        return raw

    def _parse_json(
        self,
        role: RoleSpec,
        raw: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> Any:
        """Parse a JSON reply, re-asking once with a wider budget if it was cut off.

        A reply that ends mid-string is the decoder hitting ``max_tokens``, not a model that cannot
        follow the schema — and because these roles decode greedily, asking again identically
        reproduces the same truncated bytes. A shot whose plan is one token too long therefore never
        heals: it fails, is retried, and fails at exactly the same character, holding a card for as
        long as anything keeps retrying it. Widening the budget is the only retry that can differ.
        """

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if not raw.strip():
                raise
        wider = replace(role.sampling, max_tokens=role.sampling.max_tokens * 4)
        raw = self._transport_for(role).chat(
            model=self._model_for(role), messages=messages,
            sampling=wider, schema=schema, timeout=self.timeout,
        )
        self.calls.append({"role": role.id, "key": role.key, "model": self._model_for(role),
                           "retry": "widened_budget"})
        return json.loads(raw)


# Convenience: a mock transport for tests / offline wiring checks.
@dataclass(slots=True)
class ScriptedTransport:
    """Returns canned replies; records requests. ``reply`` may be a str or a
    callable(model, messages, sampling, schema) -> str."""

    reply: Any
    seen: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.seen is None:
            self.seen = []

    def chat(self, *, model, messages, sampling, schema, timeout) -> str:  # noqa: ANN001
        self.seen.append({"model": model, "messages": messages, "sampling": sampling, "schema": schema})
        r: Any = self.reply
        if isinstance(r, Callable):  # type: ignore[arg-type]
            return r(model, messages, sampling, schema)
        return r


__all__ = [
    "Transport",
    "HttpTransport",
    "ScriptedTransport",
    "MllmRoleRunner",
    "image_content_part",
    "DEFAULT_BASE_URL",
    "DEFAULT_VLM_BASE_URL",
]
