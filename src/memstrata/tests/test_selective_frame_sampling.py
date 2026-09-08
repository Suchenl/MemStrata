from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from memstrata.bank import AssetBank
from memstrata.lib import media
from memstrata.lib.media import FrameSample
from memstrata.skills.crop_acquisition import crop_client
from memstrata.skills.crop_acquisition.crop_client import ProposeIdentifyCropper
from memstrata.skills.decomposition.decomposer import RoleAwareDecomposer


def test_selective_decoder_requests_exact_indices_and_stops_after_last(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 120
        pixels = bytes([20, 0, 0] * 4) + bytes([80, 0, 0] * 4)
        return SimpleNamespace(stdout=pixels)

    monkeypatch.setattr(media, "_video_binary", lambda name, env_name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    images = media._selective_rgb_frames(
        "segment.mp4",
        width=2,
        height=2,
        indices=[8, 2, 8],
    )

    assert images is not None
    assert sorted(images) == [2, 8]
    assert images[2].getpixel((0, 0)) == (20, 0, 0)
    assert images[8].getpixel((0, 0)) == (80, 0, 0)
    command = calls[0]
    assert command[command.index("-vf") + 1] == r"select=eq(n\,2)+eq(n\,8)"
    assert command[command.index("-frames:v") + 1] == "2"


def test_materializer_preserves_namer_and_cropper_index_rules(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[int] = []

    monkeypatch.setattr(media, "_video_shape_and_count", lambda video: (2, 2, 10))

    def fake_selective(video, *, width, height, indices):
        del video, width, height
        captured.extend(indices)
        return {
            index: Image.new("RGB", (2, 2), (index, 0, 0))
            for index in set(indices)
        }

    monkeypatch.setattr(media, "_selective_rgb_frames", fake_selective)
    namer = tmp_path / "namer.png"
    crop = tmp_path / "crop.jpg"

    saved = media.materialize_video_frames(
        "segment.mp4",
        [
            FrameSample(0.8, namer, basis="count"),
            FrameSample(0.8, crop, basis="last"),
        ],
    )

    assert captured == [8, 7]
    assert saved == [namer, crop]
    assert Image.open(namer).getpixel((0, 0)) == (8, 0, 0)


def test_production_sampler_combines_namer_and_crop_requests(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[list[FrameSample]] = []

    def fake_materialize(video, samples):
        assert video == "segment.mp4"
        calls.append(samples)
        for sample in samples:
            sample.output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), (1, 2, 3)).save(sample.output)
        return [sample.output for sample in samples]

    monkeypatch.setattr(crop_client, "materialize_video_frames", fake_materialize)
    cropper = ProposeIdentifyCropper(
        AssetBank(),
        server_dir=tmp_path / "server",
        work_dir=tmp_path / "work",
        auto_start=False,
    )

    namer_paths = cropper.sample_namer_frames(
        "segment.mp4",
        segment_id=4,
        out_dir=tmp_path / "namer",
        count=3,
        prefix="seg00004",
    )
    crop_paths, positions = cropper.frame_paths_for_segment("segment.mp4", segment_id=4)

    assert len(calls) == 1
    assert [sample.basis for sample in calls[0]] == [
        "last",
        "last",
        "last",
        "count",
        "count",
        "count",
    ]
    assert len(namer_paths) == 3
    assert len(crop_paths) == 3
    assert positions == [0.2, 0.5, 0.8]


def test_decomposer_uses_optional_shared_sampler(tmp_path) -> None:
    expected = [str(tmp_path / "frame-0.png")]

    class SharedCropper:
        def sample_namer_frames(self, segment_video: str, **kwargs):
            assert segment_video == "segment.mp4"
            assert kwargs["segment_id"] == 9
            assert kwargs["count"] == 3
            return expected

    decomposer = RoleAwareDecomposer(
        cropper=SharedCropper(),
        namer_frame_dir=tmp_path,
    )

    assert decomposer._namer_frames("segment.mp4", segment_id=9) == expected
