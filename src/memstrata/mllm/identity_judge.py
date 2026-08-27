"""VLM-first write-path identity adjudication (paper §4.5, slow write path).

The write path decides "is this discovered observation the SAME entity as an existing
record?" Historically that was a pure encoder-cosine + Dice-text score χ against a
per-type floor β_τ (see ``MemoryUpdater.identity_score``). A controlled 600-pair
robustness study (``the original calibration workspace/20260725_vlm_vs_embedding_robustness``)
showed those thresholds are brittle under occlusion/blur/low-light: the generic encoder
is the weak link and calibrated floors drift. This module adds the VLM adjudicator used
in the χ *gray zone*, keeping the fast read path and the encoder short-circuit intact.

Contract:

* ``judge(crop_a, crop_b, ...) -> IdentityVerdict`` answers same/different for two crops
  of the SAME type. ``same is None`` means *abstain* — the caller must fall back to the
  deterministic encoder threshold, never guess.
* The VLM runs at temperature 0 so identity decisions stay reproducible (the paper's
  key invariant). Per-model confidence calibration (θ) lives in the caller's policy
  (``MemoryPolicy.identity_vlm_theta``), not here: the study found 8B needs a very
  conservative merge θ while 32B is healthy at θ≈0.90.
* The offline default is :class:`NullIdentityJudge` (always abstains), so importing or
  building the write path without a configured VLM changes NO behavior.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
# The robustness study recommends 32B as the safer VLM-first default (8B's Youden
# operating point makes blur false-merge explode); override via env for a local endpoint.
DEFAULT_MODEL = "Qwen3-VL-32B-Instruct"


@dataclass(slots=True)
class IdentityVerdict:
    """Same/different judgment for two crops. ``same is None`` == abstain (fall back)."""

    same: bool | None
    confidence: float = 0.0
    source: str = "unknown"
    reasoning: str = ""


IDENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["same", "confidence", "reasoning"],
    "additionalProperties": False,
}

JUDGE_PROMPT = (
    "You are given a QUERY image of a {kind} (the first image) followed by {n} REFERENCE "
    "image(s) that all show ONE existing {kind}. Decide whether the query shows the SAME "
    "specific {kind} (same individual identity / same physical object) as the reference "
    "set, not merely the same category. The references are different views/states of the "
    "same entity, so judge against them jointly and ignore pose, expression, lighting, "
    "viewpoint, aging, or wear.\n"
    "For a character: same person (face and stable identifying features). For a prop/"
    "location: the same physical object/place.\n"
    "same: true only if you are confident the query is the same specific entity as the "
    "reference set; false if it is a different entity.\n"
    "confidence: 0..1, your calibrated probability that 'same' is correct.\n"
    "reasoning: one short clause citing the identifying evidence.\n"
    "Return JSON only."
)


def _as_reference_list(references: str | list[str]) -> list[str]:
    return [references] if isinstance(references, str) else [r for r in references if r]


class IdentityJudge(Protocol):
    def judge(
        self,
        crop: str,
        references: str | list[str],
        *,
        kind: str = "",
        name_a: str = "",
        name_b: str = "",
    ) -> IdentityVerdict: ...


class NullIdentityJudge:
    """Always abstains — the offline default; callers fall back to the encoder threshold."""

    def judge(
        self,
        crop: str,
        references: str | list[str],
        *,
        kind: str = "",
        name_a: str = "",
        name_b: str = "",
    ) -> IdentityVerdict:
        _ = crop, references, kind, name_a, name_b
        return IdentityVerdict(same=None, confidence=0.0, source="null")


class HeuristicIdentityJudge:
    """Deterministic filename-based judge for offline tests (no server).

    The query is "same" iff its filename stem shares the leading identity token (before the
    first ``@`` or ``__`` separator, e.g. ``elias@s001`` vs ``elias@s009``) with ANY of the
    reference crops. Confidence 0.99 sits above any reasonable θ so tests exercise merge.
    """

    def judge(
        self,
        crop: str,
        references: str | list[str],
        *,
        kind: str = "",
        name_a: str = "",
        name_b: str = "",
    ) -> IdentityVerdict:
        def _tok(p: str) -> str:
            stem = Path(p).stem.lower()
            for sep in ("@", "__"):
                if sep in stem:
                    stem = stem.split(sep, 1)[0]
            return stem

        refs = _as_reference_list(references)
        query_tok = _tok(crop)
        same = any(_tok(r) == query_tok for r in refs)
        return IdentityVerdict(
            same=same, confidence=0.99, source="heuristic", reasoning="stem_identity_token"
        )


def _image_data_url(image_path: str) -> str:
    import io

    from memstrata.lib.media import load_crop_rgb_for_model

    rgb = load_crop_rgb_for_model(image_path)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class VlmIdentityJudge:
    """Same/different adjudication via an OpenAI-compatible multimodal endpoint (temp 0)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout_sec: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MEMSTRATA_IDENTITY_JUDGE_BASE_URL")
            or os.environ.get("MEMSTRATA_CROP_ATTR_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.model = (
            model
            or os.environ.get("MEMSTRATA_IDENTITY_JUDGE_MODEL")
            or DEFAULT_MODEL
        )
        self.timeout_sec = timeout_sec

    def judge(
        self,
        crop: str,
        references: str | list[str],
        *,
        kind: str = "",
        name_a: str = "",
        name_b: str = "",
    ) -> IdentityVerdict:
        try:
            url_query = _image_data_url(crop)
        except OSError:
            return IdentityVerdict(same=None, source="vlm_error", reasoning="unreadable_query")
        ref_urls: list[str] = []
        for ref in _as_reference_list(references):
            try:
                ref_urls.append(_image_data_url(ref))
            except OSError:
                continue  # skip an unreadable reference, judge on the rest
        if not ref_urls:
            return IdentityVerdict(same=None, source="vlm_error", reasoning="no_readable_reference")

        prompt = JUDGE_PROMPT.format(kind=kind or "entity", n=len(ref_urls))
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url_query}},
        ]
        for url in ref_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [{"role": "user", "content": content}]
        try:
            result = self._call_api(messages)
        except Exception as exc:  # noqa: BLE001 — never crash the write path on a judge error
            return IdentityVerdict(same=None, source="vlm_error", reasoning=str(exc)[:200])

        same = result.get("same")
        return IdentityVerdict(
            same=bool(same) if isinstance(same, bool) else None,
            confidence=float(result.get("confidence", 0.0) or 0.0),
            source="vlm",
            reasoning=str(result.get("reasoning", "")),
        )

    def _call_api(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": int(os.environ.get("MEMSTRATA_IDENTITY_JUDGE_MAX_TOKENS") or "1024"),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "identity_verdict",
                    "schema": IDENTITY_SCHEMA,
                    "strict": True,
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
            return json.loads(content)


def build_identity_judge(
    *,
    mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> IdentityJudge:
    """Env-driven factory. Default ``null`` (abstain) → write-path behavior unchanged."""
    chosen = (
        mode or os.environ.get("MEMSTRATA_IDENTITY_JUDGE") or "null"
    ).strip().lower()
    if chosen in {"vlm", "mllm", "api"}:
        return VlmIdentityJudge(base_url=base_url, model=model)
    if chosen in {"heuristic", "stub", "test"}:
        return HeuristicIdentityJudge()
    return NullIdentityJudge()


__all__ = [
    "IDENTITY_SCHEMA",
    "JUDGE_PROMPT",
    "IdentityJudge",
    "IdentityVerdict",
    "NullIdentityJudge",
    "HeuristicIdentityJudge",
    "VlmIdentityJudge",
    "build_identity_judge",
]
