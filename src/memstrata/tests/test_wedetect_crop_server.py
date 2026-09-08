from __future__ import annotations

from types import SimpleNamespace


def test_healthy_wedetect_path_does_not_require_sam3(monkeypatch) -> None:
    import memstrata.skills.crop_acquisition.crop_server as crop_server
    import memstrata.skills.crop_acquisition.embedding as embedding
    import memstrata.skills.crop_acquisition.grounding_dino as grounding_dino
    import memstrata.skills.crop_acquisition.wedetect_client as wedetect_client

    class FakeDetector:
        def __init__(self, *, device=None):
            self.device = device

        def _ensure_loaded(self):
            return None

    class FakeEmbedder:
        def __init__(self, *, device=None):
            self.device = device

        def _ensure_loaded(self):
            return None

    monkeypatch.setattr(grounding_dino, "GroundingDinoProposer", FakeDetector)
    monkeypatch.setattr(embedding, "DinoV3Embedder", FakeEmbedder)
    monkeypatch.setattr(
        wedetect_client.WeDetectRefGrounder,
        "from_env",
        classmethod(lambda cls, *, required=False: object()),
    )

    models = crop_server._Models(device="cpu")

    assert models.grounder is not None
    assert models.segmenter is None
    assert models.detector is not None
    assert models.embedder is not None


def test_server_removes_its_ready_sentinel_on_idle_exit(monkeypatch, tmp_path) -> None:
    import memstrata.skills.crop_acquisition.crop_server as crop_server

    monkeypatch.setattr(crop_server, "_Models", lambda **_kwargs: object())
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(crop_server.time, "time", lambda: next(ticks))
    server_dir = tmp_path / "server"

    crop_server.serve(
        SimpleNamespace(server_dir=str(server_dir), device="", idle_timeout=1.0)
    )

    assert not (server_dir / "ready").exists()
