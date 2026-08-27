from __future__ import annotations


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
        classmethod(lambda cls: object()),
    )

    models = crop_server._Models(device="cpu")

    assert models.grounder is not None
    assert models.segmenter is None
    assert models.detector is not None
    assert models.embedder is not None
