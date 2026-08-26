"""Memory bank for MemStrata (paper §3): stratified records m_j = (τ, e, R, d, ψ)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AssetType(str, Enum):
    """τ_j ∈ {character, prop, location} (Table: production memory types)."""

    CHARACTER = "character"
    PROP = "prop"
    LOCATION = "location"


# Backward-compatible alias used by older call sites / scripts.
AssetKind = AssetType


class LifecycleStatus(str, Enum):
    """ψ_j (paper §3 Step 4)."""

    CANDIDATE = "candidate"
    REUSABLE = "reusable"
    USED = "used"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    FAILED = "failed"


# Alias for older call sites.
AssetStatus = LifecycleStatus

NON_USABLE = frozenset({
    LifecycleStatus.REJECTED,
    LifecycleStatus.DEPRECATED,
    LifecycleStatus.FAILED,
})


class AssetVersionConflictError(RuntimeError):
    """A curation proposal was built from an obsolete bank snapshot."""


class SpatialAngle(str, Enum):
    """Spatial view of a visual evidence crop (MemStrata visual stratum)."""

    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    TOP = "top"
    UNKNOWN = "unknown"


class StateAngle(str, Enum):
    """Appearance / condition state of a visual evidence crop."""

    DEFAULT = "default"
    CHANGED = "changed"
    DAMAGED = "damaged"
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    REPLACES = "replaces"
    DEPRECATED_BY = "deprecated_by"


@dataclass(slots=True)
class AssetRelation:
    relation_type: RelationType
    target_asset_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssetRepresentation:
    """Visual evidence r ∈ R_j as a stratified [image + angle] record."""

    representation_id: str
    asset_id: str
    object_uri: str
    origin_segment_id: int = 0
    representation_type: str = "image_set"
    # Visual stratum: each crop carries explicit spatial / temporal / state angles.
    spatial_angle: SpatialAngle = SpatialAngle.UNKNOWN
    state_angle: StateAngle = StateAngle.UNKNOWN
    # temporal angle is the origin segment (and optional free-form tag).
    temporal_tag: str = ""
    reference_aspects: list[str] = field(default_factory=list)  # α⁺
    excluded_aspects: list[str] = field(default_factory=list)  # α⁻
    quality_by_purpose: dict[str, float] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    deprecated_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "asset_id": self.asset_id,
            "object_uri": self.object_uri,
            "origin_segment_id": self.origin_segment_id,
            "representation_type": self.representation_type,
            "spatial_angle": self.spatial_angle.value,
            "state_angle": self.state_angle.value,
            "temporal_tag": self.temporal_tag,
            "reference_aspects": self.reference_aspects,
            "excluded_aspects": self.excluded_aspects,
            "quality_by_purpose": self.quality_by_purpose,
            "annotations": self.annotations,
            "deprecated": self.deprecated,
            "deprecated_by": self.deprecated_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetRepresentation:
        spatial = str(data.get("spatial_angle", SpatialAngle.UNKNOWN.value))
        state = str(data.get("state_angle", StateAngle.UNKNOWN.value))
        try:
            spatial_angle = SpatialAngle(spatial)
        except ValueError:
            spatial_angle = SpatialAngle.UNKNOWN
        try:
            state_angle = StateAngle(state)
        except ValueError:
            state_angle = StateAngle.UNKNOWN
        return cls(
            representation_id=data["representation_id"],
            asset_id=data["asset_id"],
            object_uri=data["object_uri"],
            origin_segment_id=int(data.get("origin_segment_id", 0)),
            representation_type=str(data.get("representation_type", "image_set")),
            spatial_angle=spatial_angle,
            state_angle=state_angle,
            temporal_tag=str(data.get("temporal_tag", "")),
            reference_aspects=list(data.get("reference_aspects", [])),
            excluded_aspects=list(data.get("excluded_aspects", [])),
            quality_by_purpose={
                str(key): float(value)
                for key, value in dict(data.get("quality_by_purpose", {})).items()
            },
            annotations=dict(data.get("annotations", {})),
            deprecated=bool(data.get("deprecated", False)),
            deprecated_by=str(data.get("deprecated_by", "")),
        )


@dataclass(slots=True)
class Asset:
    """Memory record m_j = (τ_j, e_j, R_j, d_j, ψ_j)."""

    asset_id: str  # e_j
    kind: AssetType  # τ_j
    name: str
    status: LifecycleStatus  # ψ_j
    description: str = ""  # d_j
    representations: list[AssetRepresentation] = field(default_factory=list)  # R_j
    relations: list[AssetRelation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> str:
        return self.description or str(self.metadata.get("description", ""))


class AssetBank:
    """Persistent memory bank M_n: O(1) lookup by identifier."""

    def __init__(self, assets: dict[str, Asset] | None = None, *, version: int = 0) -> None:
        self.assets: dict[str, Asset] = assets or {}
        self.version = max(0, int(version))

    def touch(self) -> int:
        """Advance the in-memory snapshot version after an accepted mutation."""
        self.version += 1
        return self.version

    def commit_snapshot(self, snapshot: dict[str, Any], *, expected_version: int) -> int:
        """Atomically replace this in-memory bank when its baseline still matches."""
        if self.version != expected_version:
            raise AssetVersionConflictError(
                f"stale asset snapshot: expected {expected_version}, current {self.version}"
            )
        restored = type(self).from_dict(snapshot)
        self.assets = restored.assets
        self.version = max(self.version + 1, restored.version)
        return self.version

    def add_asset(self, asset: Asset) -> None:
        self.assets[asset.asset_id] = asset
        self.touch()

    def get_asset(self, asset_id: str) -> Asset | None:
        return self.assets.get(asset_id)

    def remove_asset(self, asset_id: str) -> None:
        if asset_id in self.assets:
            self.assets.pop(asset_id)
            self.touch()

    def list_assets(
        self,
        kind: AssetType | None = None,
        status: LifecycleStatus | None = None,
    ) -> list[Asset]:
        out = list(self.assets.values())
        if kind is not None:
            out = [a for a in out if a.kind == kind]
        if status is not None:
            out = [a for a in out if a.status == status]
        return out

    def usable_assets(self) -> list[Asset]:
        return [a for a in self.assets.values() if a.status not in NON_USABLE]

    def update_status(self, asset_id: str, status: LifecycleStatus) -> None:
        asset = self.get_asset(asset_id)
        if asset is not None and asset.status != status:
            asset.status = status
            self.touch()

    def find_by_name(self, name: str, kind: AssetType | None = None) -> Asset | None:
        """Name-anchored lookup (axiom 3: high aggregation).

        Matches the canonical ``name`` first, then any declared alias in
        ``metadata['aliases']`` so a single identity referred to by different
        surface forms ("the priest" / "Father Janovich") aggregates into one
        asset instead of splitting. Aliases are *explicit only* — no fuzzy
        auto-merge — keeping identity name-anchored and reproducible. This
        mirrors the read path (``memory_retrieval.name_match``), which already
        consults aliases, so write and read stay symmetric.
        """
        key = name.strip().lower()
        if not key:
            return None
        canonical: Asset | None = None
        aliased: Asset | None = None
        for asset in self.assets.values():
            if kind is not None and asset.kind != kind:
                continue
            if asset.name.strip().lower() == key:
                canonical = asset
                break
            if aliased is None:
                aliases = asset.metadata.get("aliases") or []
                if isinstance(aliases, list) and any(
                    str(alias).strip().lower() == key for alias in aliases
                ):
                    aliased = asset
        return canonical if canonical is not None else aliased

    def register_alias(self, asset_id: str, alias: str) -> bool:
        """Record a surface-name variant for an asset. Returns True iff added.

        Lets upstream naming resolve variants to one identity (axiom 3) without
        creating duplicate assets. The canonical name is never treated as an alias.
        """
        asset = self.get_asset(asset_id)
        alias_key = str(alias or "").strip()
        if asset is None or not alias_key:
            return False
        if alias_key.strip().lower() == asset.name.strip().lower():
            return False
        aliases = asset.metadata.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        if any(str(a).strip().lower() == alias_key.lower() for a in aliases):
            return False
        aliases.append(alias_key)
        asset.metadata["aliases"] = aliases
        self.touch()
        return True

    def find_representation(self, rep_id: str) -> tuple[Asset, AssetRepresentation] | None:
        for asset in self.assets.values():
            for rep in asset.representations:
                if rep.representation_id == rep_id:
                    return asset, rep
        return None

    def deprecated_representation_ids(self) -> list[str]:
        ids = [
            rep.representation_id
            for asset in self.assets.values()
            for rep in asset.representations
            if rep.deprecated
        ]
        return sorted(ids)

    def to_dict(self) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for aid, asset in self.assets.items():
            serialized[aid] = {
                "asset_id": asset.asset_id,
                "kind": asset.kind.value,
                "name": asset.name,
                "status": asset.status.value,
                "description": asset.description,
                "representations": [rep.to_dict() for rep in asset.representations],
                "relations": [
                    {
                        "relation_type": relation.relation_type.value,
                        "target_asset_id": relation.target_asset_id,
                        "attributes": relation.attributes,
                    }
                    for relation in asset.relations
                ],
                "metadata": asset.metadata,
            }
        return {"version": self.version, "assets": serialized}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetBank:
        assets: dict[str, Asset] = {}
        for aid, raw in data.get("assets", {}).items():
            kind_raw = str(raw["kind"])
            try:
                kind = AssetType(kind_raw)
            except ValueError:
                continue  # drop non-paper types from legacy dumps
            status_raw = str(raw["status"])
            try:
                status = LifecycleStatus(status_raw)
            except ValueError:
                status = LifecycleStatus.REUSABLE
            description = str(raw.get("description", "") or raw.get("metadata", {}).get("description", ""))
            metadata = dict(raw.get("metadata", {}))
            if description and "description" not in metadata:
                metadata["description"] = description
            assets[aid] = Asset(
                asset_id=raw["asset_id"],
                kind=kind,
                name=raw["name"],
                status=status,
                description=description,
                representations=[AssetRepresentation.from_dict(r) for r in raw.get("representations", [])],
                relations=[
                    AssetRelation(
                        relation_type=RelationType(item["relation_type"]),
                        target_asset_id=str(item["target_asset_id"]),
                        attributes=dict(item.get("attributes", {})),
                    )
                    for item in raw.get("relations", [])
                    if item.get("relation_type") in RelationType._value2member_map_
                    and item.get("target_asset_id")
                ],
                metadata=metadata,
            )
        return cls(assets=assets, version=int(data.get("version", 0)))

    def save(self, path: str | Path) -> None:
        """Atomically persist the bank as JSON (write-temp then rename)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> AssetBank:
        """Restore a bank previously written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Backward-compatible name from earlier drafts / scripts.
ProductionAssetSpace = AssetBank
