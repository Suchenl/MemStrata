"""Model-free dedup over embeddings: identity matching and non-redundant selection."""

from __future__ import annotations

import re

from memstrata.encoders.base import Vector, cosine_distance, cosine_similarity


def cosine_or_none(left: Vector, right: Vector) -> float | None:
    """Cosine similarity, or ``None`` when the two vectors are not comparable.

    Guards the write path against silently comparing embeddings from different
    encoders. The bare ``zip``-based dot product this replaces truncated to the
    shorter vector and returned a meaningless score, which made near-duplicate
    suppression fail invisibly as soon as one rep was embedded by a different
    route (e.g. a 64-d fallback seed vs a 512-d face embedding).
    """
    if not left or not right or len(left) != len(right):
        return None
    return cosine_similarity(left, right)


_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def text_tokens(text: str) -> set[str]:
    """Lowercased content tokens; CJK as overlapping bigrams (search-engine style).

    English / numeric tokens keep the whole word (>= 3 chars); CJK runs are emitted as
    overlapping character bigrams (a single character for length-1 runs), mirroring the
    read-side interpreter tokenizer (``intent_understanding._content_tokens``) so write-side
    dedup and read-side description matching judge Chinese text the same way.

    The earlier per-character CJK split was too loose: two descriptions sharing a single
    incidental character (红色 / 红衣 share 红) scored as near-duplicates. English
    tokenization is UNCHANGED, so the LSMDC (English control set) reconcile-threshold
    calibration is unaffected — only CJK matching tightens.
    """
    out: set[str] = set()
    s = str(text or "").casefold()
    for token in _ASCII_TOKEN_RE.findall(s):
        if len(token) >= 3:
            out.add(token)
    for run in _CJK_RUN_RE.findall(s):
        if len(run) == 1:
            out.add(run)
            continue
        for i in range(len(run) - 1):
            out.add(run[i:i + 2])
    return out


def text_similarity(left: str, right: str) -> float:
    """Deterministic Dice coefficient over content tokens, in ``[0, 1]``.

    Model-free stand-in for ``sim^text`` in the identity-reconciliation score
    (paper Eq. identity_score). Deterministic on purpose: identity decisions must
    be reproducible across runs, and a text encoder would make the write path
    depend on a served model.
    """
    a, b = text_tokens(left), text_tokens(right)
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def match_to_existing(
    query: Vector,
    candidates: list[tuple[str, Vector]],
    *,
    threshold: float = 0.6,
) -> tuple[str | None, float]:
    """Return ``(best_id, score)`` if the closest candidate is within ``threshold``."""

    best_id: str | None = None
    best_score = -1.0
    for identifier, vector in candidates:
        score = cosine_similarity(query, vector)
        if score > best_score:
            best_score = score
            best_id = identifier
    if best_id is not None and best_score >= threshold:
        return best_id, best_score
    return None, best_score


def similarity_to_set(query: Vector, references: list[Vector]) -> float:
    """Max cosine similarity of ``query`` to any reference; ``-1.0`` if none.

    Admission-consistency helper (design_philosophy.md §2 gate ②): a new crop that
    is far from *every* existing visible representation of an asset is a likely
    mislabel / intruder — it verifies WHO before WHERE, without reassigning identity.
    """
    best = -1.0
    for vector in references:
        score = cosine_or_none(query, vector)
        if score is None:
            continue  # different encoder route → not comparable, not "dissimilar"
        best = max(best, score)
    return best


def medoid_cohesion(vectors: list[Vector]) -> tuple[int, list[float], float]:
    """Return ``(medoid_index, sims_to_medoid, min_pairwise)`` for a rep cluster.

    The medoid is the vector with the highest average similarity to the others; it
    is a robust "center" for per-asset identity-cohesion self-audit. ``min_pairwise``
    is the smallest pairwise similarity in the cluster — a low value means the asset's
    representations do not form one identity (mixed / name-collision). For 0 or 1
    vector the cluster is trivially cohesive (``min_pairwise = 1.0``).
    """
    count = len(vectors)
    if count == 0:
        return -1, [], 1.0
    if count == 1:
        return 0, [1.0], 1.0

    sims: list[list[float]] = [[0.0] * count for _ in range(count)]
    min_pairwise = 1.0
    for i in range(count):
        for j in range(i + 1, count):
            score = cosine_similarity(vectors[i], vectors[j])
            sims[i][j] = score
            sims[j][i] = score
            min_pairwise = min(min_pairwise, score)

    medoid = max(range(count), key=lambda i: sum(sims[i]) / (count - 1))
    return medoid, sims[medoid], min_pairwise


def largest_cohesive_subcluster(
    vectors: list[Vector],
    *,
    link_threshold: float,
) -> list[int]:
    """Return sorted indices of the largest mutually-cohesive subcluster.

    Builds an undirected graph whose edges connect any two vectors with cosine
    similarity ``>= link_threshold``, then returns the largest connected component.
    This is a robust proxy for the "majority identity mass" of an asset's
    representations. Ties break toward higher average internal similarity, then
    lower first index.

    Used as an *alternative* self-audit reference to the single ``medoid`` (see
    ``MemoryUpdater.audit_cohesion`` / design_philosophy §2.1). On labelled S5 data the
    two are tied when intruders are a minority, but the subcluster is meaningfully
    stronger when an asset is *majority-polluted* (the medoid can then land on an
    intruder), so it is kept as an opt-in fallback for high-mixing regimes.
    """
    count = len(vectors)
    if count == 0:
        return []
    if count == 1:
        return [0]

    parent = list(range(count))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(count):
        for j in range(i + 1, count):
            if cosine_similarity(vectors[i], vectors[j]) >= link_threshold:
                _union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(count):
        components.setdefault(_find(i), []).append(i)

    def _avg_internal(idxs: list[int]) -> float:
        if len(idxs) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                total += cosine_similarity(vectors[idxs[a]], vectors[idxs[b]])
                pairs += 1
        return total / pairs

    best = max(
        components.values(),
        key=lambda idxs: (len(idxs), _avg_internal(idxs), -min(idxs)),
    )
    return sorted(best)


def select_non_redundant(
    vectors: list[Vector],
    *,
    max_keep: int = 5,
    min_distance: float = 0.15,
    quality: list[float] | None = None,
) -> list[int]:
    """Greedy quality-seeded farthest-point selection."""

    count = len(vectors)
    if count == 0:
        return []
    if quality is not None and len(quality) != count:
        raise ValueError("quality must align with vectors")
    if max_keep <= 0:
        return []

    order = sorted(range(count), key=lambda i: (-(quality[i] if quality else 0.0), i))
    kept: list[int] = [order[0]]
    for index in order[1:]:
        if len(kept) >= max_keep:
            break
        nearest = min(cosine_distance(vectors[index], vectors[chosen]) for chosen in kept)
        if nearest >= min_distance:
            kept.append(index)
    return sorted(kept)


def select_attribute_diverse(
    *,
    bucket_keys: list[tuple[str, ...]],
    vectors: list[Vector] | None = None,
    quality: list[float] | None = None,
    max_keep: int = 5,
    min_distance: float = 0.15,
    pin: list[int] | None = None,
) -> list[int]:
    """Prefer distinct attribute buckets, then embedding diversity.

    ``bucket_keys[i]`` is typically ``(spatial, state, shot_size, lighting)``.
    Known (non-``unknown``) labels in a bucket outrank all-unknown buckets.
    Within a bucket, keep the highest-quality index. Remaining slots are filled
    via farthest-point on embeddings when provided.

    ``pin`` lists indices that must survive and are charged against ``max_keep``
    before anything else. Callers use it to reserve a slot for evidence that is
    valuable for a reason this function cannot see — notably the newest
    observation, which otherwise loses every tie-break to already-stored reps and
    can never enter a full asset.
    """

    count = len(bucket_keys)
    if count == 0:
        return []
    if quality is not None and len(quality) != count:
        raise ValueError("quality must align with bucket_keys")
    if vectors is not None and len(vectors) != count:
        raise ValueError("vectors must align with bucket_keys")
    if max_keep <= 0:
        return []

    def _quality(i: int) -> float:
        return float(quality[i]) if quality else 0.0

    def _known(i: int) -> int:
        return sum(1 for part in bucket_keys[i] if part and part != "unknown")

    kept: list[int] = []
    for index in pin or []:
        if 0 <= index < count and index not in kept and len(kept) < max_keep:
            kept.append(index)
    pinned = set(kept)

    best_for_bucket: dict[tuple[str, ...], int] = {}
    for index in range(count):
        if index in pinned:
            continue
        key = bucket_keys[index]
        prev = best_for_bucket.get(key)
        if prev is None:
            best_for_bucket[key] = index
            continue
        if (_known(index), _quality(index), -index) > (_known(prev), _quality(prev), -prev):
            best_for_bucket[key] = index

    bucket_winners = sorted(
        best_for_bucket.values(),
        key=lambda i: (-_known(i), -_quality(i), i),
    )
    for index in bucket_winners:
        if len(kept) >= max_keep:
            break
        kept.append(index)

    if len(kept) >= max_keep or vectors is None:
        return sorted(kept)

    remaining = [i for i in range(count) if i not in kept]
    remaining.sort(key=lambda i: (-_quality(i), i))
    for index in remaining:
        if len(kept) >= max_keep:
            break
        sims = [
            s
            for s in (cosine_or_none(vectors[index], vectors[chosen]) for chosen in kept)
            if s is not None
        ]
        # Incomparable embeddings must not silently read as "far apart"; without a
        # usable similarity the candidate cannot claim a diversity slot.
        if sims and (1.0 - max(sims)) >= min_distance:
            kept.append(index)
    return sorted(kept)


def compatible_stratum(
    *,
    rep_bucket: tuple[str, ...],
    new_bucket: tuple[str, ...],
) -> bool:
    """Whether two attribute buckets describe a comparable visual stratum.

    Implements ``compat(r, ŝ, b̂)`` from the paper's redundancy definition: an
    observation is only redundant against evidence in the *same* stratum, so a new
    view or state is never discarded for looking unlike a stored one. An
    ``unknown`` label on either side cannot prove the strata differ, so it stays
    compatible — which makes this a no-op (all reps compatible) whenever the
    attribute classifier is disabled, preserving the offline default.
    """
    for mine, theirs in zip(rep_bucket, new_bucket):
        if not mine or not theirs or mine == "unknown" or theirs == "unknown":
            continue
        if mine != theirs:
            return False
    return True


def select_angle_diverse(
    *,
    spatial_angles: list[str],
    state_angles: list[str],
    vectors: list[Vector] | None = None,
    quality: list[float] | None = None,
    max_keep: int = 5,
    min_distance: float = 0.15,
) -> list[int]:
    """Prefer distinct ``(spatial, state)`` buckets, then embedding diversity.

    Known angles outrank ``unknown``. Within a bucket, keep the highest-quality
    index. Remaining slots (if any) are filled via farthest-point on embeddings.
    """

    count = len(spatial_angles)
    if count == 0:
        return []
    if len(state_angles) != count:
        raise ValueError("state_angles must align with spatial_angles")
    bucket_keys = list(zip(spatial_angles, state_angles, strict=True))
    return select_attribute_diverse(
        bucket_keys=bucket_keys,
        vectors=vectors,
        quality=quality,
        max_keep=max_keep,
        min_distance=min_distance,
    )
