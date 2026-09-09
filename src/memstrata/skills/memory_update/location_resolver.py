"""Conservative, causal proposals for resolving location observations.

This first-stage resolver is intentionally shadow-only: it records what a safe
location-specific resolver *would* do without changing the current asset identity.
Later storage work can enforce these proposals once multi-view place evidence and a
multi-valued location name index are available.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from memstrata.bank.schema import surface_key


class LocationSemanticRelation(str, Enum):
    STRICT_SYNONYM = "strict_synonym"
    PART_OF = "part_of"
    ADJACENT_TO = "adjacent_to"
    INTERIOR_OF = "interior_of"
    UNKNOWN = "unknown"


class LocationResolutionAction(str, Enum):
    REUSE = "reuse"
    MERGE_ALIAS = "merge_alias"
    RELATE = "relate"
    NEW_ASSET = "new_asset"
    DEFER = "defer"


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return min(1.0, max(-1.0, float(value)))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class LocationResolutionEvidence:
    """Evidence supplied by existing naming/place/continuity passes.

    ``semantic_relation`` is an upstream assertion, not inferred from benchmark
    vocabulary here.  Only names actually observed by the writer can become aliases.
    """

    candidate_asset_id: str = ""
    semantic_relation: LocationSemanticRelation = LocationSemanticRelation.UNKNOWN
    visual_similarity: float | None = None
    temporally_continuous: bool = False
    independent_support: int = 0
    encoder_route: str = ""

    @classmethod
    def from_mapping(cls, raw: Any) -> LocationResolutionEvidence:
        row = raw if isinstance(raw, Mapping) else {}
        relation_raw = str(row.get("semantic_relation") or "unknown")
        try:
            relation = LocationSemanticRelation(relation_raw)
        except ValueError:
            relation = LocationSemanticRelation.UNKNOWN
        try:
            support = max(0, int(row.get("independent_support", 0) or 0))
        except (TypeError, ValueError):
            support = 0
        return cls(
            candidate_asset_id=str(row.get("candidate_asset_id") or ""),
            semantic_relation=relation,
            visual_similarity=_optional_float(row.get("visual_similarity")),
            temporally_continuous=bool(row.get("temporally_continuous", False)),
            independent_support=support,
            encoder_route=str(row.get("encoder_route") or ""),
        )


@dataclass(frozen=True, slots=True)
class LocationResolverPolicy:
    min_visual_similarity: float = 0.80
    min_independent_support: int = 2
    allow_continuity_support: bool = True
    max_temporal_gap: int = 8


@dataclass(frozen=True, slots=True)
class LocationResolutionProposal:
    action: LocationResolutionAction
    candidate_asset_id: str
    relation_type: str | None
    observed_alias: str | None
    reasons: tuple[str, ...]
    evidence: LocationResolutionEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "candidate_asset_id": self.candidate_asset_id,
            "relation_type": self.relation_type,
            "observed_alias": self.observed_alias,
            "reasons": list(self.reasons),
            "evidence": {
                **asdict(self.evidence),
                "semantic_relation": self.evidence.semantic_relation.value,
            },
            "shadow_only": True,
        }


@dataclass(frozen=True, slots=True)
class LocationRelationProposal:
    """Auditable structural relation inferred without changing either identity."""

    action: LocationResolutionAction
    candidate_asset_id: str
    relation_type: str | None
    source_role: str | None
    target_role: str | None
    read_neighbor: bool
    reasons: tuple[str, ...]
    evidence: LocationResolutionEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "memstrata.location-relation-proposal.v1",
            "action": self.action.value,
            "candidate_asset_id": self.candidate_asset_id,
            "relation_type": self.relation_type,
            "source_role": self.source_role,
            "target_role": self.target_role,
            "read_neighbor": self.read_neighbor,
            "reasons": list(self.reasons),
            "evidence": {
                **asdict(self.evidence),
                "semantic_relation": self.evidence.semantic_relation.value,
            },
        }


_STRUCTURAL_RELATIONS = frozenset(
    {
        LocationSemanticRelation.PART_OF,
        LocationSemanticRelation.ADJACENT_TO,
        LocationSemanticRelation.INTERIOR_OF,
    }
)


def _lexically_contains_place(parent_name: str, child_name: str) -> bool:
    """Return whether ``child`` is a qualified surface containing ``parent``.

    This is deliberately syntax-only.  It can propose ``part_of`` but is never
    identity evidence.  Token containment handles spaced languages; conservative
    prefix/suffix containment handles punctuation-free CJK names.
    """

    parent = surface_key(parent_name)
    child = surface_key(child_name)
    if not parent or parent == child or len(parent) >= len(child):
        return False
    parent_tokens = parent.split()
    child_tokens = child.split()
    if len(child_tokens) > 1 and len(parent_tokens) <= len(child_tokens):
        width = len(parent_tokens)
        return any(
            child_tokens[index : index + width] == parent_tokens
            for index in range(len(child_tokens) - width + 1)
        )
    # A one-character CJK overlap is too weak to establish even a read neighbor.
    return len(parent) >= 2 and (child.startswith(parent) or child.endswith(parent))


def propose_lexical_location_relation(
    *,
    incoming_name: str,
    candidate_name: str,
    evidence: LocationResolutionEvidence,
    policy: LocationResolverPolicy | None = None,
) -> LocationRelationProposal | None:
    """Propose a causal ``part_of`` edge for qualified location surfaces.

    The edge is safe for candidate expansion only when both endpoints have
    independent scene evidence and occur in one configured temporal neighborhood.
    Otherwise the same lexical observation is retained as a deferred audit record.
    """

    pol = policy or LocationResolverPolicy()
    incoming_is_child = _lexically_contains_place(candidate_name, incoming_name)
    candidate_is_child = _lexically_contains_place(incoming_name, candidate_name)
    if not incoming_is_child and not candidate_is_child:
        return None
    confirmed = (
        evidence.temporally_continuous
        and evidence.independent_support >= pol.min_independent_support
    )
    return LocationRelationProposal(
        action=(
            LocationResolutionAction.RELATE
            if confirmed
            else LocationResolutionAction.DEFER
        ),
        candidate_asset_id=evidence.candidate_asset_id,
        relation_type=LocationSemanticRelation.PART_OF.value,
        source_role="incoming" if incoming_is_child else "candidate",
        target_role="candidate" if incoming_is_child else "incoming",
        read_neighbor=confirmed,
        reasons=(
            ("qualified_surface", "short_range_continuity", "scene_evidence")
            if confirmed
            else ("qualified_surface_unconfirmed",)
        ),
        evidence=evidence,
    )


def propose_location_resolution(
    *,
    incoming_name: str,
    candidate_name: str = "",
    evidence: LocationResolutionEvidence | None = None,
    policy: LocationResolverPolicy | None = None,
) -> LocationResolutionProposal:
    """Propose reuse/alias/relation/new without mutating the bank.

    Exact location names are not identity-authoritative here.  They still require
    visual plus repeated or continuous evidence, because generic names such as
    ``forest`` may refer to several physical places.
    """

    ev = evidence or LocationResolutionEvidence()
    pol = policy or LocationResolverPolicy()
    relation = ev.semantic_relation

    if relation in _STRUCTURAL_RELATIONS:
        return LocationResolutionProposal(
            LocationResolutionAction.RELATE,
            ev.candidate_asset_id,
            relation.value,
            None,
            ("structural_relation_not_identity",),
            ev,
        )

    if not ev.candidate_asset_id or not candidate_name:
        return LocationResolutionProposal(
            LocationResolutionAction.NEW_ASSET,
            ev.candidate_asset_id,
            None,
            None,
            ("no_candidate_anchor",),
            ev,
        )

    visual_ok = (
        ev.visual_similarity is not None
        and ev.visual_similarity >= pol.min_visual_similarity
    )
    support_ok = ev.independent_support >= pol.min_independent_support
    if pol.allow_continuity_support and ev.temporally_continuous:
        support_ok = True
    confirmed = visual_ok and support_ok

    if relation is LocationSemanticRelation.STRICT_SYNONYM:
        if confirmed:
            return LocationResolutionProposal(
                LocationResolutionAction.MERGE_ALIAS,
                ev.candidate_asset_id,
                None,
                incoming_name,
                ("strict_synonym", "visual_match", "causal_support"),
                ev,
            )
        return LocationResolutionProposal(
            LocationResolutionAction.DEFER,
            ev.candidate_asset_id,
            None,
            None,
            ("strict_synonym_unconfirmed",),
            ev,
        )

    exact_name = bool(surface_key(incoming_name)) and (
        surface_key(incoming_name) == surface_key(candidate_name)
    )
    if exact_name and confirmed:
        return LocationResolutionProposal(
            LocationResolutionAction.REUSE,
            ev.candidate_asset_id,
            None,
            None,
            ("surface_match", "visual_match", "causal_support"),
            ev,
        )
    if exact_name:
        return LocationResolutionProposal(
            LocationResolutionAction.DEFER,
            ev.candidate_asset_id,
            None,
            None,
            ("same_name_not_identity_evidence",),
            ev,
        )

    return LocationResolutionProposal(
        LocationResolutionAction.NEW_ASSET,
        ev.candidate_asset_id,
        None,
        None,
        ("different_unrelated_surface",),
        ev,
    )


__all__ = [
    "LocationResolutionAction",
    "LocationResolutionEvidence",
    "LocationResolutionProposal",
    "LocationRelationProposal",
    "LocationResolverPolicy",
    "LocationSemanticRelation",
    "propose_lexical_location_relation",
    "propose_location_resolution",
]
