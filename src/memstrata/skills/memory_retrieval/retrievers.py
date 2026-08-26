"""Self-contained frame/segment retrieval variants for controlled ablations."""

from __future__ import annotations

import math
import os
import random
import re
import subprocess
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

# Shared encoder substrate (same normal import other skills use, e.g. memory_update /
# embedding_deduplication). No file-path hack: the skill references the shared foundation
# through the package API, so it stays migratable without reaching into a sibling's source.
from memstrata.encoders import (
    EmbeddingModel,
    TextEmbeddingModel,
    Vector,
    build_image_embedding,
    build_text_embedding,
    cosine_similarity,
)


@dataclass(slots=True)
class RetrievedRef:
    score: float
    arm: str
    asset_id: str | None = None
    representation_id: str | None = None
    source_seconds: float | None = None


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        ...


@dataclass(slots=True)
class RetrieverConfig:
    """Runtime knobs shared by the retrieval ablations."""

    text_provider: str = "hash"
    frame_provider: str = "hash"
    keyframe_provider: str = "hash"
    text_model: str | None = None
    frame_model: str | None = None
    keyframe_model: str | None = None
    uniform_fps: float = 1.0
    dense_fps: float = 2.0
    key_per_segment: int = 3
    topk_segment: int = 5
    ffmpeg: str = "ffmpeg"
    extract_frames: bool = True
    random_seed: int = 0

    @classmethod
    def from_env(cls) -> "RetrieverConfig":
        return cls(
            text_provider=os.environ.get("RETR_TEXT_PROVIDER", "hash"),
            frame_provider=os.environ.get("RETR_FRAME_PROVIDER", "hash"),
            keyframe_provider=os.environ.get("RETR_KEYFRAME_PROVIDER", "hash"),
            text_model=os.environ.get("RETR_TEXT_MODEL") or None,
            frame_model=os.environ.get("RETR_FRAME_MODEL") or None,
            keyframe_model=os.environ.get("RETR_KEYFRAME_MODEL") or None,
            uniform_fps=float(os.environ.get("RETR_UNIFORM_FPS", "1.0")),
            dense_fps=float(os.environ.get("RETR_DENSE_FPS", "2.0")),
            key_per_segment=int(os.environ.get("RETR_KEY_PER_SEG", "3")),
            topk_segment=int(os.environ.get("RETR_TOPK_SEG", "5")),
            ffmpeg=os.environ.get("RETR_FFMPEG", "ffmpeg"),
            extract_frames=os.environ.get("RETR_EXTRACT_FRAMES", "1") != "0",
            random_seed=int(os.environ.get("RETR_RANDOM_SEED", "0")),
        )


@dataclass(slots=True)
class FrameRecord:
    frame_id: str
    segment_id: int
    source_seconds: float
    image_ref: str
    text_vector: Vector | None = None
    frame_vector: Vector | None = None
    key_vector: Vector | None = None


@dataclass(slots=True)
class SegmentRecord:
    segment_id: int
    prompt_text: str
    seconds_span: tuple[float, float]
    description: str = ""
    text_vector: Vector | None = None
    uniform_frames: dict[float, list[FrameRecord]] = field(default_factory=dict)
    dense_frames: dict[float, list[FrameRecord]] = field(default_factory=dict)


def _content_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", str(text or "").casefold())


def _embed_query(model: TextEmbeddingModel, text: str) -> Vector:
    fn = getattr(model, "embed_query", None)
    return fn(text) if callable(fn) else model.embed_text(text)


def _embed_doc(model: TextEmbeddingModel, text: str) -> Vector:
    fn = getattr(model, "embed_doc", None)
    return fn(text) if callable(fn) else model.embed_text(text)


def _ref_from_frame(frame: FrameRecord, *, score: float, arm: str) -> RetrievedRef:
    return RetrievedRef(
        score=float(score),
        arm=arm,
        representation_id=frame.frame_id,
        source_seconds=float(frame.source_seconds),
    )


class MemoryRetrievalStore:
    """Historical segment/frame memory shared by the retrieval variants."""

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig.from_env()
        self.movie_id = ""
        self.source_video = ""
        self.work_dir = Path(".")
        self.segments: list[SegmentRecord] = []
        self.frames: list[FrameRecord] = []
        self._text: TextEmbeddingModel | None = None
        self._frame_embedder: EmbeddingModel | None = None
        self._key_embedder: EmbeddingModel | None = None

    def reset(self, *, movie_id: str, source_video: str, work_dir: str | Path) -> None:
        self.movie_id = movie_id
        self.source_video = source_video
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.segments = []
        self.frames = []
        self._text = None
        self._frame_embedder = None
        self._key_embedder = None

    def observe_segment(
        self,
        *,
        segment_id: int,
        prompt_text: str,
        seconds_span: tuple[float, float],
        segment_video: str = "",
        description: str = "",
    ) -> None:
        del segment_video  # source-video seconds are enough for frame materialization.
        seg = SegmentRecord(
            segment_id=int(segment_id),
            prompt_text=str(prompt_text or ""),
            seconds_span=(float(seconds_span[0]), float(seconds_span[1])),
            description=str(description or prompt_text or ""),
        )
        self.segments.append(seg)

    def _text_model(self) -> TextEmbeddingModel:
        if self._text is None:
            self._text = build_text_embedding(
                provider=self.config.text_provider,
                model=self.config.text_model,
            )
        return self._text

    def _frame_model(self) -> EmbeddingModel:
        if self._frame_embedder is None:
            self._frame_embedder = build_image_embedding(
                provider=self.config.frame_provider,
                model=self.config.frame_model,
            )
        return self._frame_embedder

    def _key_model(self) -> EmbeddingModel:
        if self._key_embedder is None:
            self._key_embedder = build_image_embedding(
                provider=self.config.keyframe_provider,
                model=self.config.keyframe_model,
            )
        return self._key_embedder

    def eligible_segments(self, as_of_seconds: float) -> list[SegmentRecord]:
        return [s for s in self.segments if float(s.seconds_span[1]) <= float(as_of_seconds)]

    def eligible_frames(self, as_of_seconds: float) -> list[FrameRecord]:
        return [f for f in self.frames if float(f.source_seconds) < float(as_of_seconds)]

    def _sample_times(self, span: tuple[float, float], fps: float) -> list[float]:
        s0, s1 = float(span[0]), float(span[1])
        dur = max(0.0, s1 - s0)
        if dur <= 0.0:
            return [s0]
        fps = max(float(fps), 1e-6)
        step = 1.0 / fps
        times: list[float] = []
        t = s0 + min(step * 0.5, dur * 0.5)
        while t < s1 - 1e-6:
            times.append(round(t, 3))
            t += step
        if not times:
            times.append(round(s0 + dur * 0.5, 3))
        return times

    def _frame_path(self, seg: SegmentRecord, seconds: float) -> Path:
        return self.work_dir / "retrieval_frames" / f"c{seg.segment_id:05d}_{seconds:.3f}.png"

    def _maybe_extract_frame(self, seg: SegmentRecord, seconds: float) -> str:
        out = self._frame_path(seg, seconds)
        if out.is_file() and out.stat().st_size > 0:
            return str(out)
        if not self.config.extract_frames or not self.source_video or not Path(self.source_video).is_file():
            return f"{self.movie_id}:c{seg.segment_id}:{seconds:.3f}"
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                self.config.ffmpeg,
                "-y",
                "-ss",
                f"{seconds:.3f}",
                "-i",
                self.source_video,
                "-frames:v",
                "1",
                "-vf",
                "scale=512:-1",
                "-q:v",
                "2",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            return str(out)
        return f"{self.movie_id}:c{seg.segment_id}:{seconds:.3f}"

    def frames_for_segment(self, seg: SegmentRecord, *, fps: float, dense: bool = False) -> list[FrameRecord]:
        cache = seg.dense_frames if dense else seg.uniform_frames
        key = round(float(fps), 6)
        cached = cache.get(key)
        if cached is not None:
            return cached
        out: list[FrameRecord] = []
        for seconds in self._sample_times(seg.seconds_span, fps):
            frame_id = f"c{seg.segment_id:05d}@{seconds:.3f}"
            ref = self._maybe_extract_frame(seg, seconds)
            rec = FrameRecord(
                frame_id=frame_id,
                segment_id=seg.segment_id,
                source_seconds=seconds,
                image_ref=ref,
            )
            out.append(rec)
            self.frames.append(rec)
        cache[key] = out
        return out

    def rank_segments_text(
        self, query: str, *, as_of_seconds: float, limit: int | None = None
    ) -> list[tuple[SegmentRecord, float]]:
        model = self._text_model()
        q = _embed_query(model, query)
        ranked: list[tuple[SegmentRecord, float]] = []
        for seg in self.eligible_segments(as_of_seconds):
            text = seg.description or seg.prompt_text
            if not text:
                continue
            if seg.text_vector is None:
                seg.text_vector = _embed_doc(model, text)
            ranked.append((seg, cosine_similarity(q, seg.text_vector)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:limit] if limit is not None else ranked

    def rank_frames_text(
        self,
        query: str,
        *,
        as_of_seconds: float,
        frames: list[FrameRecord] | None = None,
        limit: int | None = None,
    ) -> list[tuple[FrameRecord, float]]:
        model = self._frame_model()
        text_model = model if hasattr(model, "embed_text") else self._text_model()
        q = _embed_query(text_model, query)  # type: ignore[arg-type]
        pool = frames if frames is not None else self.eligible_frames(as_of_seconds)
        ranked: list[tuple[FrameRecord, float]] = []
        for frame in pool:
            if frame.source_seconds >= float(as_of_seconds):
                continue
            if frame.frame_vector is None:
                frame.frame_vector = model.embed_image(frame.image_ref)
            ranked.append((frame, cosine_similarity(q, frame.frame_vector)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:limit] if limit is not None else ranked

    def keyframes_for_segment(self, seg: SegmentRecord, *, per_segment: int) -> list[FrameRecord]:
        frames = self.frames_for_segment(seg, fps=self.config.dense_fps, dense=True)
        if len(frames) <= per_segment:
            return frames
        model = self._key_model()
        for frame in frames:
            if frame.key_vector is None:
                frame.key_vector = model.embed_image(frame.image_ref)
        selected = [frames[0]]
        remaining = frames[1:]
        while remaining and len(selected) < per_segment:
            best_idx = 0
            best_dist = -1.0
            for idx, cand in enumerate(remaining):
                assert cand.key_vector is not None
                nearest = max(
                    cosine_similarity(cand.key_vector, sel.key_vector or cand.key_vector)
                    for sel in selected
                )
                dist = 1.0 - nearest
                if dist > best_dist:
                    best_dist = dist
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return sorted(selected, key=lambda f: f.source_seconds)


class _BaseRetriever:
    name = "retriever"

    def __init__(self, store: MemoryRetrievalStore) -> None:
        self.store = store

    def _budget(self, budget: int) -> int:
        return max(0, int(budget))


class SegUniformRetriever(_BaseRetriever):
    name = "seg_uniform_ablation"

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        refs: list[RetrievedRef] = []
        segs = self.store.rank_segments_text(
            query, as_of_seconds=as_of_seconds, limit=self.store.config.topk_segment
        )
        for seg_rank, (seg, score) in enumerate(segs, start=1):
            frames = self.store.frames_for_segment(seg, fps=self.store.config.uniform_fps)
            for frame_rank, frame in enumerate(frames, start=1):
                refs.append(_ref_from_frame(
                    frame,
                    score=float(score) - 1e-6 * (seg_rank + frame_rank),
                    arm="segment_text_uniform",
                ))
        refs.sort(key=lambda r: r.score, reverse=True)
        return refs[: self._budget(budget)]


class SegDinoKeyRetriever(_BaseRetriever):
    name = "seg_dinokey_ablation"

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        refs: list[RetrievedRef] = []
        segs = self.store.rank_segments_text(
            query, as_of_seconds=as_of_seconds, limit=self.store.config.topk_segment
        )
        for seg_rank, (seg, score) in enumerate(segs, start=1):
            frames = self.store.keyframes_for_segment(seg, per_segment=self.store.config.key_per_segment)
            for frame_rank, frame in enumerate(frames, start=1):
                refs.append(_ref_from_frame(
                    frame,
                    score=float(score) - 1e-6 * (seg_rank + frame_rank),
                    arm="segment_text_dinokey",
                ))
        refs.sort(key=lambda r: r.score, reverse=True)
        return refs[: self._budget(budget)]


class FrameTextRetriever(_BaseRetriever):
    name = "frame_text_ablation"

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        # Ensure every observed segment contributes sampled frames to the frame pool.
        for seg in self.store.eligible_segments(as_of_seconds):
            self.store.frames_for_segment(seg, fps=self.store.config.uniform_fps)
        ranked = self.store.rank_frames_text(query, as_of_seconds=as_of_seconds, limit=budget)
        return [_ref_from_frame(frame, score=score, arm="frame_text") for frame, score in ranked]


def _rrf_rankings(rankings: list[list[RetrievedRef]], *, k: int = 60) -> list[RetrievedRef]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievedRef] = {}
    for ranking in rankings:
        for rank, ref in enumerate(ranking, start=1):
            key = ref.representation_id or f"{ref.source_seconds:.3f}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best.setdefault(key, ref)
    fused: list[RetrievedRef] = []
    for key, score in scores.items():
        ref = best[key]
        fused.append(RetrievedRef(
            score=score,
            arm="rrf",
            asset_id=ref.asset_id,
            representation_id=ref.representation_id,
            source_seconds=ref.source_seconds,
        ))
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


class SegFrameRerankRetriever(_BaseRetriever):
    name = "seg_framererank_ablation"

    def rankings(self, query: str, *, as_of_seconds: float, budget: int) -> list[list[RetrievedRef]]:
        seg_ranked = self.store.rank_segments_text(
            query, as_of_seconds=as_of_seconds, limit=self.store.config.topk_segment
        )
        candidate_frames: list[FrameRecord] = []
        coarse_refs: list[RetrievedRef] = []
        for seg_rank, (seg, _score) in enumerate(seg_ranked, start=1):
            frames = self.store.frames_for_segment(seg, fps=self.store.config.uniform_fps)
            candidate_frames.extend(frames)
            for frame in frames:
                coarse_refs.append(_ref_from_frame(
                    frame, score=1.0 / seg_rank, arm="coarse_segment_rank"
                ))
        fine_ranked = self.store.rank_frames_text(
            query,
            as_of_seconds=as_of_seconds,
            frames=candidate_frames,
            limit=max(budget * 4, budget),
        )
        fine_refs = [_ref_from_frame(frame, score=score, arm="fine_frame_rank") for frame, score in fine_ranked]
        return [coarse_refs, fine_refs]

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        return _rrf_rankings(self.rankings(query, as_of_seconds=as_of_seconds, budget=budget))[
            : self._budget(budget)
        ]


class RecencyRetriever(_BaseRetriever):
    name = "recency_ctrl"

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        del query
        for seg in self.store.eligible_segments(as_of_seconds):
            self.store.frames_for_segment(seg, fps=self.store.config.uniform_fps)
        frames = sorted(
            self.store.eligible_frames(as_of_seconds), key=lambda f: f.source_seconds, reverse=True
        )
        refs = []
        for rank, frame in enumerate(frames[: self._budget(budget)], start=1):
            refs.append(_ref_from_frame(frame, score=1.0 / rank, arm="recency"))
        return refs


class BM25DescriptionRetriever(_BaseRetriever):
    name = "bm25_desc_ctrl"

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        segs = self.store.eligible_segments(as_of_seconds)
        docs = [s.description or s.prompt_text for s in segs]
        q_tokens = _content_tokens(query)
        doc_tokens = [_content_tokens(doc) for doc in docs]
        if not q_tokens or not any(doc_tokens):
            return RecencyRetriever(self.store).retrieve(query, as_of_seconds=as_of_seconds, budget=budget)
        n_docs = len(docs)
        avgdl = sum(len(toks) for toks in doc_tokens) / max(1, n_docs)
        df: dict[str, int] = {}
        for toks in doc_tokens:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        k1, b = 1.5, 0.75
        scored: list[tuple[SegmentRecord, float]] = []
        for seg, toks in zip(segs, doc_tokens):
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            score = 0.0
            dl = len(toks)
            for tok in q_tokens:
                freq = tf.get(tok, 0)
                if not freq:
                    continue
                idf = math.log(1.0 + (n_docs - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5))
                denom = freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6))
                score += idf * (freq * (k1 + 1.0) / denom)
            if score > 0.0:
                scored.append((seg, score))
        if not scored:
            return RecencyRetriever(self.store).retrieve(query, as_of_seconds=as_of_seconds, budget=budget)
        scored.sort(key=lambda x: x[1], reverse=True)
        refs: list[RetrievedRef] = []
        for seg, score in scored:
            frame = self.store.frames_for_segment(seg, fps=self.store.config.uniform_fps)[0]
            refs.append(_ref_from_frame(frame, score=score, arm="bm25_desc"))
        return refs[: self._budget(budget)]


class RandomRetriever(_BaseRetriever):
    name = "random_ctrl"

    def retrieve(self, query: str, *, as_of_seconds: float, budget: int) -> list[RetrievedRef]:
        for seg in self.store.eligible_segments(as_of_seconds):
            self.store.frames_for_segment(seg, fps=self.store.config.uniform_fps)
        frames = self.store.eligible_frames(as_of_seconds)
        seed_material = f"{self.store.config.random_seed}:{query}:{as_of_seconds}".encode("utf-8")
        seed = int.from_bytes(sha256(seed_material).digest()[:8], "big")
        rng = random.Random(seed)
        rng.shuffle(frames)
        refs: list[RetrievedRef] = []
        for rank, frame in enumerate(frames[: self._budget(budget)], start=1):
            refs.append(_ref_from_frame(frame, score=1.0 / rank, arm="random"))
        return refs


_BUILDERS: dict[str, type[_BaseRetriever]] = {
    "seg_uniform": SegUniformRetriever,
    "seg_uniform_ablation": SegUniformRetriever,
    "seg_dinokey": SegDinoKeyRetriever,
    "seg_dinokey_ablation": SegDinoKeyRetriever,
    "seg_framererank": SegFrameRerankRetriever,
    "seg_framererank_ablation": SegFrameRerankRetriever,
    "frame_text": FrameTextRetriever,
    "frame_text_ablation": FrameTextRetriever,
    "recency": RecencyRetriever,
    "recency_ctrl": RecencyRetriever,
    "bm25_desc": BM25DescriptionRetriever,
    "bm25_desc_ctrl": BM25DescriptionRetriever,
    "random": RandomRetriever,
    "random_ctrl": RandomRetriever,
}


def build_retriever(
    variant: str,
    *,
    store: MemoryRetrievalStore | None = None,
    config: RetrieverConfig | None = None,
) -> Retriever:
    key = (variant or "seg_framererank").strip().lower()
    cls = _BUILDERS.get(key)
    if cls is None:
        raise ValueError(f"unknown retrieval variant {variant!r}; choices: {sorted(_BUILDERS)}")
    return cls(store or MemoryRetrievalStore(config))
