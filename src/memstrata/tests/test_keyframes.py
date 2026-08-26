from pathlib import Path

from memstrata.skills.crop_acquisition.keyframes import select_diverse_keyframes


class _FakeEmbedder:
    """Returns preset unit vectors keyed by frame filename stem index."""

    def __init__(self, vecs: dict[str, list[float]]) -> None:
        self.vecs = vecs
        self.calls = 0

    def embed_batch(self, images: list[Path]) -> list[list[float]]:
        self.calls += 1
        return [self.vecs[Path(p).name] for p in images]


def _paths(*names: str) -> list[str]:
    return [f"/tmp/{n}" for n in names]


def test_fps_picks_representative_plus_diverse_drops_duplicates():
    # frames a/b/c are near-duplicates ([1,0]); d and e are distinct directions.
    vecs = {
        "a.png": [1.0, 0.0],
        "b.png": [1.0, 0.0],
        "c.png": [1.0, 0.0],
        "d.png": [0.0, 1.0],
        "e.png": [-1.0, 0.0],
    }
    emb = _FakeEmbedder(vecs)
    picked = select_diverse_keyframes(_paths("a.png", "b.png", "c.png", "d.png", "e.png"), emb, k=3)
    assert emb.calls == 1
    names = {Path(p).name for p in picked}
    assert len(picked) == 3
    # representative seed (a, highest mean sim) comes first
    assert Path(picked[0]).name == "a.png"
    # the two distinct views are chosen over the a/b/c duplicates
    assert "d.png" in names and "e.png" in names
    assert not ({"b.png", "c.png"} & names)


def test_returns_all_when_fewer_than_k():
    emb = _FakeEmbedder({})
    picked = select_diverse_keyframes(_paths("a.png", "b.png"), emb, k=5)
    assert picked == _paths("a.png", "b.png")
    assert emb.calls == 0  # no need to embed


def test_empty_and_nonpositive_k():
    emb = _FakeEmbedder({})
    assert select_diverse_keyframes([], emb, k=3) == []
    assert select_diverse_keyframes(_paths("a.png"), emb, k=0) == []


def test_embed_failure_falls_back_to_first_k():
    class _Boom:
        def embed_batch(self, images):
            raise RuntimeError("no gpu")

    picked = select_diverse_keyframes(_paths("a.png", "b.png", "c.png", "d.png"), _Boom(), k=2)
    assert picked == _paths("a.png", "b.png")
