"""Materialize Composed Context → controls['composed_references']."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memstrata.bank import AssetBank
from memstrata.steps.compose import ComposedContext

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def composed_reference_images(
    composed: ComposedContext,
    bank: AssetBank,
) -> list[dict[str, Any]]:
    """Resolve selected assets to still-image paths for reference-conditioned backends.

    Returns ``[{asset_id, kind, name, role, image}]`` in selection order.
    Dedupes by absolute image path.
    """
    refs: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for asset_id in composed.asset_ids:
        asset = bank.get_asset(asset_id)
        if asset is None:
            continue
        has_explicit_selection = asset_id in composed.representation_ids
        chosen = list(composed.representation_ids.get(asset_id, []))
        reps = asset.representations
        if has_explicit_selection:
            by_id = {rep.representation_id: rep for rep in reps}
            reps = [by_id[rep_id] for rep_id in chosen if rep_id in by_id]
        for rep in reps:
            if rep.deprecated:
                continue
            uri = rep.object_uri
            if not uri:
                continue
            path = Path(str(uri))
            if not (path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES):
                continue
            image = str(path.resolve())
            if image in seen_images:
                continue
            seen_images.add(image)
            role = composed.functions.get(asset_id) or (
                rep.reference_aspects[0] if rep.reference_aspects else "reference"
            )
            refs.append({
                "asset_id": asset_id,
                "kind": str(asset.kind.value if hasattr(asset.kind, "value") else asset.kind),
                "name": asset.name,
                "role": role,
                "image": image,
            })
            # Legacy callers that did not provide explicit representation ids retain
            # the historical one-image fallback. Explicit adaptive selections materialize
            # every chosen representative in their ranked order.
            if not has_explicit_selection:
                break
    return refs


def reference_directives(composed: ComposedContext, bank: AssetBank) -> list[dict[str, Any]]:
    """Directive list for MediaGenerationTask.reference_directives."""
    out: list[dict[str, Any]] = []
    for asset_id in composed.asset_ids:
        asset = bank.get_asset(asset_id)
        if asset is None:
            continue
        out.append({
            "asset_id": asset_id,
            "kind": asset.kind.value,
            "name": asset.name,
            "function": composed.functions.get(asset_id, "identity_anchor"),
            "requirement": composed.requirements.get(asset_id, "continuity"),
            "representation_ids": list(composed.representation_ids.get(asset_id, [])),
            "strength": "required",
        })
    return out
