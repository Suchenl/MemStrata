"""VLM-based decomposition backend (write path) — MemStrata's own multimodal model
proposes the typed named entities present in a realized video segment.

This is the paper's Evidence-Acquisition write front-end done by the method's *own* VLM:
given one or more frames sampled from the realized segment together with the user's
generation prompt for that segment, the model lists the salient entities that are
actually visible, each typed (character / prop / location) with a label, a short English
``category`` common noun (used as the open-vocabulary segmenter concept — a generic
"object" concept finds no props, a specific "red apple" does), and a short visual
description. Naming follows the paper's requested/discovered split: an entity the
prompt refers to by name is labelled with that name (the user's intended, SUT-visible
name for a *requested* asset), while an entity absent from the prompt gets a descriptive
label and stays a *discovered* candidate. Names are bound only to entities visually
confirmed in the frames, and are never read from benchmark annotations (present sets,
entity registry, roster). Cross-segment identity is still decided downstream by
``curate`` reconciliation, not asserted here.

Because a VLM asked to describe frames tends to drift from the prompt's surface form (it
may answer '紫色小鸟' as 'purple bird'), which breaks the deterministic read-side name
recall, ``propose`` runs a name-reconciliation pass: any proposed label the deterministic
matcher cannot find in the prompt is sent to a SINGLE fresh-context auditor call (a
separate model turn — not a self-review of the generation that produced it) that returns
the prompt's exact wording, which is then re-verified against the prompt before being
accepted. See ``VlmEntityDecomposer._reconcile_names``.

It reuses the shared MLLM transport (:class:`memstrata.mllm.runner.MllmRoleRunner`,
role ``entity_decomposer``), so it needs no HTTP code of its own and is unit-testable
against a ``ScriptedTransport``. On any server/parse failure it returns ``[]`` so the
caller can fall back to the perception (SAM3-concept) proposer without failing the segment.
"""

from __future__ import annotations

from typing import Any

from memstrata.bank import AssetType
from memstrata.skills.decomposition.decomposer import NamedEntity

_KIND_BY_STR: dict[str, AssetType] = {
    "character": AssetType.CHARACTER,
    "prop": AssetType.PROP,
    "location": AssetType.LOCATION,
}

_INSTRUCTION = (
    "You are MemStrata's visual memory analyst. You are given one or more frames sampled "
    "from a SINGLE generated video segment, plus the user's generation prompt for that "
    "segment. List the salient, memory-worthy entities that are ACTUALLY VISIBLE in the "
    "frames. For each entity give:\n"
    "  - kind: exactly one of character | prop | location. A living being (person OR any "
    "animal — rabbit, bird, squirrel, butterfly, insect, ...) is ALWAYS 'character', never "
    "'prop'. 'prop' is for inanimate objects (fruit, tool, rope, ...); 'location' is the "
    "setting/place.\n"
    "  - label: the entity's name. If the prompt refers to this entity, copy the prompt's "
    "exact wording for it VERBATIM, in the prompt's own language and script — do NOT "
    "translate, transliterate, or paraphrase it (e.g. if the prompt says '紫色小鸟', the "
    "label MUST be '紫色小鸟', never 'purple bird'). Only if the entity is absent from the "
    "prompt, use a concise, specific descriptive label, written in the SAME language as the "
    "prompt. Never attach a prompt name to an entity that is not actually visible in the "
    "frames, and never invent a proper name that is neither visible nor stated in the prompt.\n"
    "  - category: a SHORT ENGLISH common noun naming what this entity concretely IS, usable "
    "as an open-vocabulary segmentation concept (e.g. 'red apple', 'acorn', 'vine rope', "
    "'rabbit', 'butterfly', 'meadow'). Lowercase, 1-3 words, singular, no proper names — this "
    "drives the segmenter, so make it the most specific concrete object noun that fits.\n"
    "  - description: one clause of concrete visual appearance (clothing, colors, shape, "
    "material) grounded in the frames, enough to re-identify the entity later.\n"
    "Rules: only list what is visible in the frames; do not list entities the prompt "
    "mentions but that do not appear; at most one 'location' (the setting/place). Do NOT "
    "list non-entities: backgrounds, the plain sky, end-credits / title cards / on-screen "
    "text / subtitles, logos, watermarks, or generic undifferentiated scenery — only "
    "concrete, re-identifiable characters, props, and the single setting. Return "
    "at most {max_entities} entities, most important first."
)

# Deterministic non-entity stoplist (fix: bank pollution). A VLM asked for "memory-worthy
# entities" still occasionally emits backgrounds / end credits / on-screen text as if they
# were assets; these are never re-identifiable entities and only bloat the bank, so they are
# dropped before entities are proposed regardless of the model's judgement. Kept small and
# conservative on purpose — only obvious non-entities, matched on the label OR the segmenter
# category so a drifted synonym is still caught.
_STOP_EXACT: frozenset[str] = frozenset(
    {
        "background", "backdrop", "screen", "screens", "credits", "end credits",
        "end credit", "title", "title card", "titles", "logo", "watermark", "blank",
        "empty", "scenery", "nothing", "sky",
    }
)
_STOP_SUBSTR_CJK: tuple[str, ...] = (
    "背景", "屏幕", "片尾", "片头", "字幕", "水印", "标题", "空白", "片头曲", "片尾曲",
)


def _is_noise_label(label: str, category: str) -> bool:
    """True for obvious non-entities (backgrounds / credits / on-screen text)."""
    lab = label.strip().lower()
    cat = category.strip().lower()
    if lab in _STOP_EXACT or (cat and cat in _STOP_EXACT):
        return True
    for token in _STOP_SUBSTR_CJK:
        if token in label or (category and token in category):
            return True
    return False

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["character", "prop", "location"]},
                    "label": {"type": "string"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["kind", "label", "category", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

# Fresh-context naming auditor. Runs ONLY on labels the deterministic name matcher could
# not find in the prompt (e.g. the model translated '紫色小鸟' to 'purple bird'), in a
# SEPARATE model call — never a self-review inside the generation that produced the label —
# and re-binds each to the prompt's exact surface form so the read-side fast name path can
# recall it. Model claims are still re-verified deterministically before being accepted.
_RECONCILE_INSTRUCTION = (
    "You are a naming auditor. You did NOT write the labels below; review them fresh. You "
    "are given frames from one generated video segment, the user's generation prompt for it, "
    "and a numbered list of entities another model detected in the frames (each with a "
    "provisional label and a visual description). None of these labels textually appear in "
    "the prompt. For EACH one, make a single decision — is it a MIS-NAMED prompt entity, or a "
    "GENUINELY NEW entity the prompt never mentions?\n"
    "  - MIS-NAMED (the prompt does refer to this same entity, but the label paraphrased or "
    "translated it, e.g. 'purple bird' for '紫色小鸟'): set in_prompt=true and return "
    "prompt_name = the prompt's EXACT wording for it, copied verbatim in the prompt's own "
    "language and script (never translated). Use only words that actually appear in the prompt.\n"
    "  - GENUINELY NEW / self-discovered (the prompt does not name this entity at all): set "
    "in_prompt=false and echo the provisional label unchanged.\n"
    "Judge by visual identity (the description) against what the prompt says is present; never "
    "force a prompt name onto an entity the prompt never mentions.\n\n"
    "Detected entities:\n{listing}\n\nUser generation prompt:\n{prompt}"
)

_RECONCILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resolved": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "in_prompt": {"type": "boolean"},
                    "prompt_name": {"type": "string"},
                },
                "required": ["index", "in_prompt", "prompt_name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["resolved"],
    "additionalProperties": False,
}


class VlmEntityDecomposer:
    """Propose typed named entities from a realized-segment frame via the method's VLM.

    Parameters
    ----------
    runner:
        A :class:`memstrata.mllm.runner.MllmRoleRunner`. Defaults to one built from env
        (unified Qwen3.5-9B endpoint). Inject a runner with a ``ScriptedTransport`` in tests.
    max_entities:
        Hard cap on entities returned per frame (keeps the write path bounded).
    """

    def __init__(
        self,
        runner: Any | None = None,
        *,
        max_entities: int = 6,
        reconcile_names: bool = True,
    ) -> None:
        if runner is None:
            from memstrata.mllm.runner import MllmRoleRunner

            runner = MllmRoleRunner()
        self.runner = runner
        self.max_entities = int(max_entities)
        # Fresh-context name reconciliation for prompt-named entities whose label drifted
        # from the prompt's surface form (see ``_reconcile_names``).
        self.reconcile_names = bool(reconcile_names)

    def propose(self, *, frames: str | list[str], prompt: str = "") -> list[NamedEntity]:
        """Return typed named entities visible in ``frames``. Empty on failure.

        ``frames`` is one path or several frame paths sampled from the SAME realized
        segment; multiple views make naming and description more robust (a requested
        entity may only be recognizable in one of them). Capped at 8 images (endpoint
        limit). ``prompt`` is the user-visible segment prompt used to bind names to
        entities that are visually confirmed in the frames.
        """
        if isinstance(frames, str):
            frame_list = [frames] if frames else []
        else:
            frame_list = [f for f in frames if f]
        frame_list = frame_list[:8]
        if not frame_list:
            return []
        instruction = _INSTRUCTION.format(max_entities=self.max_entities)
        if prompt:
            instruction += (
                "\n\nUser generation prompt for this segment. Write EVERY label in the same "
                "language as this prompt, and for any entity this prompt names, reuse the "
                "prompt's exact words verbatim (do not translate). Use it to name the entities "
                "you can visually confirm; do not add entities that are absent from the "
                f"frames:\n{prompt}"
            )
        try:
            result = self.runner.run(
                "entity_decomposer",
                instruction=instruction,
                images=frame_list,
                schema=_SCHEMA,
            )
        except Exception:
            return []

        rows = result.get("entities") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            return []

        entities: list[NamedEntity] = []
        seen_locations = 0
        for row in rows[: self.max_entities]:
            if not isinstance(row, dict):
                continue
            kind = _KIND_BY_STR.get(str(row.get("kind", "")).strip().lower())
            if kind is None:
                continue
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            category = str(row.get("category") or "").strip()
            if _is_noise_label(label, category):
                continue  # background / credits / on-screen text are not assets
            if kind is AssetType.LOCATION:
                if seen_locations >= 1:
                    continue  # one setting per frame (the recurring place stratum)
                seen_locations += 1
            entities.append(
                NamedEntity(
                    name=label,
                    kind=kind,
                    category=category,
                    description=str(row.get("description") or "").strip(),
                )
            )
        # One call already emitted everything (matched=requested, unmatched=candidate
        # discovery); this second pass only disambiguates the unmatched bucket into
        # mis-named-prompt-entity (snap → hit) vs genuine self-discovery (keep).
        if prompt and self.reconcile_names:
            entities = self._reconcile_names(entities, frame_list, prompt)
        # Requested vs discovered split (paper Evidence Acquisition). An entity whose final
        # label is verbatim in the prompt is a *requested* asset → anchor it by that name
        # (stable symbolic id) so its reps group across segments. Everything else is a
        # *discovered* candidate → leave it UNANCHORED (entity_id=None) so cross-segment
        # identity is decided downstream by visual reconciliation (χ) against the whole
        # same-type bank, NOT by its drifting descriptive label. This is what stops the same
        # rabbit from fragmenting into 棕色兔子/棕色动物/橙色兔子/… separate records. Without a
        # prompt we cannot tell requested from discovered, so keep the historical name anchor.
        if prompt:
            from memstrata.skills.memory_retrieval.name_match import term_in_prompt

            for ent in entities:
                ent.entity_id = ent.name if term_in_prompt(prompt, ent.name) else None
        else:
            for ent in entities:
                ent.entity_id = ent.name
        return entities

    def _reconcile_names(
        self, entities: list[NamedEntity], frames: list[str], prompt: str
    ) -> list[NamedEntity]:
        """Disambiguate the *unmatched* bucket: mis-named prompt entity vs genuine discovery.

        ``propose`` emits everything in one call — entities whose label is in the prompt are
        already matched (requested), the rest are "unmatched". But that unmatched bucket mixes
        two textually-identical cases the deterministic matcher cannot tell apart: a **drift**
        (a prompt entity the VLM paraphrased/translated, e.g. 'purple bird' for '紫色小鸟')
        vs a **genuine self-discovery** (an entity the prompt never names). Only a model that
        sees the frames can decide, so the unmatched labels are sent to a SINGLE fresh-context
        auditor call — a separate model turn, not a self-review of the generation that produced
        them:

          * ``in_prompt=true``  → it was mis-named; snap ``name`` to the prompt's verbatim
            wording so it becomes a normal read-side name match (a hit).
          * ``in_prompt=false`` → it is a genuine discovery; keep the descriptive label.

        Deterministic-first: entities already matched to the prompt are never sent (0 cost). A
        snapped name is re-verified with ``term_in_prompt`` before being accepted, so a
        hallucinated name (not actually in the prompt) is rejected and the original label kept.
        Any failure leaves ``entities`` unchanged (best-effort, never fails a segment).
        """
        from memstrata.skills.memory_retrieval.name_match import term_in_prompt

        unmatched = [(i, e) for i, e in enumerate(entities) if not term_in_prompt(prompt, e.name)]
        if not unmatched:
            return entities
        listing = "\n".join(
            f"{i}. label='{e.name}' (kind={e.kind.value}; appearance: {e.description})"
            for i, e in unmatched
        )
        instruction = _RECONCILE_INSTRUCTION.format(listing=listing, prompt=prompt)
        try:
            result = self.runner.run(
                "entity_decomposer",
                instruction=instruction,
                images=frames,
                schema=_RECONCILE_SCHEMA,
            )
        except Exception:
            return entities
        rows = result.get("resolved") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            return entities
        valid_idx = {i for i, _ in unmatched}
        for row in rows:
            if not isinstance(row, dict) or not row.get("in_prompt"):
                continue
            try:
                idx = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            name = str(row.get("prompt_name") or "").strip()
            # Only accept a rename that is (a) for an entity we actually flagged and (b) a
            # term the deterministic matcher can confirm is really in the prompt.
            if idx in valid_idx and name and term_in_prompt(prompt, name):
                entities[idx].name = name
        return entities
