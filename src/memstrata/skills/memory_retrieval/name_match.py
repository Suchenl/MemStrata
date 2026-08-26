"""Name/alias-based memory retrieval (deterministic read path).

Given the current prompt and the assets already in memory, decide *which
remembered entity the prompt refers to* by matching the prompt against each
asset's stored name and aliases. This is a retrieval capability — "recall the
asset the user is talking about" — not composition logic, so it lives under
``memory_retrieval`` rather than ``composition``. ``composition.intent`` calls
into it for the fast read path (paper: name-anchored identity).

Matching is model-free and generic: an English word-boundary alternation plus a
longest-first CJK substring pass, with overlapping mentions resolved so that a
longer name wins over its shorter aliases. No lexicons, no eval-derived word
lists — only the names/aliases that the write path stored on the assets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from memstrata.bank import Asset, AssetBank


@dataclass(frozen=True, slots=True)
class NameHit:
    asset_id: str
    start: int
    end: int
    term: str


@dataclass(slots=True)
class NameMatchCache:
    version: int
    pattern: re.Pattern[str] | None
    term_assets: dict[str, list[str]]
    cjk_terms: tuple[str, ...]


# Keyed by the bank *object* (not ``id(bank)``): a plain-int id can be reused after a
# bank is GC'd, and if the reused id collides with a stale (id, version) key the wrong
# snapshot is served to a fresh bank. A WeakKeyDictionary keys on object identity and
# auto-drops entries when the bank dies, so distinct banks can never collide.
_NAME_CACHE_BY_BANK: "WeakKeyDictionary[AssetBank, tuple[int, NameMatchCache]]" = (
    WeakKeyDictionary()
)


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def term_in_prompt(prompt: str, term: str) -> bool:
    term = str(term or "").strip()
    if not term:
        return False
    if any("\u4e00" <= char <= "\u9fff" for char in term):
        # CJK one-character names are too noisy for substring matching; the indexed
        # path below enforces the same guard and longest-first overlap handling.
        return len(term) >= 2 and term.casefold() in prompt.casefold()
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
    return re.search(pattern, prompt, re.IGNORECASE) is not None


def iter_name_terms(asset: Asset) -> list[str]:
    aliases = asset.metadata.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    seen: set[str] = set()
    terms: list[str] = []
    for raw in [asset.name, *[str(alias) for alias in aliases]]:
        term = str(raw or "").strip()
        if not term:
            continue
        if has_cjk(term) and len(term) < 2:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def build_name_cache(candidates: list[Asset], *, version: int) -> NameMatchCache:
    term_assets: dict[str, list[str]] = {}
    non_cjk_terms: set[str] = set()
    cjk_terms: set[str] = set()

    for asset in candidates:
        for term in iter_name_terms(asset):
            key = term.casefold()
            term_assets.setdefault(key, []).append(asset.asset_id)
            if has_cjk(term):
                cjk_terms.add(key)
            else:
                non_cjk_terms.add(term)

    pattern = None
    if non_cjk_terms:
        ordered = sorted(non_cjk_terms, key=lambda t: (-len(t), t.casefold()))
        pattern = re.compile(
            r"(?<![A-Za-z0-9])(?:"
            + "|".join(re.escape(term) for term in ordered)
            + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
    return NameMatchCache(
        version=version,
        pattern=pattern,
        term_assets=term_assets,
        cjk_terms=tuple(sorted(cjk_terms, key=lambda t: (-len(t), t))),
    )


def match_cache(bank: AssetBank, candidates: list[Asset]) -> NameMatchCache:
    version = int(bank.version)
    entry = _NAME_CACHE_BY_BANK.get(bank)
    if entry is not None and entry[0] == version:
        return entry[1]
    cache = build_name_cache(candidates, version=version)
    # bank.version is the coherence key; a bumped version supersedes the prior snapshot.
    _NAME_CACHE_BY_BANK[bank] = (version, cache)
    return cache


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def name_hits(prompt: str, cache: NameMatchCache) -> list[NameHit]:
    lowered = prompt.casefold()
    raw_hits: list[tuple[int, int, str]] = []
    if cache.pattern is not None:
        raw_hits.extend((m.start(), m.end(), m.group(0)) for m in cache.pattern.finditer(prompt))
    for term in cache.cjk_terms:
        start = 0
        while True:
            idx = lowered.find(term, start)
            if idx < 0:
                break
            raw_hits.append((idx, idx + len(term), prompt[idx: idx + len(term)]))
            start = idx + max(1, len(term))

    # Longest-first keeps "兔子主角" over its shorter aliases; output is restored to
    # mention order so multi-entity prompts remain deterministic.
    kept_spans: list[tuple[int, int]] = []
    kept: list[tuple[int, int, str]] = []
    for start, end, term in sorted(raw_hits, key=lambda h: (-(h[1] - h[0]), h[0], h[2].casefold())):
        span = (start, end)
        if any(_overlaps(span, existing) for existing in kept_spans):
            continue
        kept_spans.append(span)
        kept.append((start, end, term))

    hits: list[NameHit] = []
    for start, end, term in sorted(kept, key=lambda h: (h[0], -(h[1] - h[0]), h[2].casefold())):
        for asset_id in cache.term_assets.get(term.casefold(), []):
            hits.append(NameHit(asset_id=asset_id, start=start, end=end, term=term))
    return hits


def unique_asset_ids(hits: list[NameHit]) -> list[str]:
    return list(dict.fromkeys(hit.asset_id for hit in hits))


def name_match(prompt: str, candidates: list[Asset]) -> list[str]:
    cache = build_name_cache(candidates, version=-1)
    return unique_asset_ids(name_hits(prompt, cache))
