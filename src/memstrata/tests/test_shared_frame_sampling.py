from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from memstrata.bank import AssetBank
from memstrata.skills.crop_acquisition.crop_client import ProposeIdentifyCropper
from memstrata.skills.decomposition.decomposer import RoleAwareDecomposer


def _pixel(path: str | Path) -> int:
    with Image.open(path) as image:
        return int(image.convert("RGB").getpixel((0, 0))[0])


def test_production_sampler_decodes_once_for_namer_and_crop_frames(
    tmp_path,
    monkeypatch,
) -> None:
    frames = np.stack(
        [np.full((4, 4, 3), index * 20, dtype=np.uint8) for index in range(10)]
    )
    cropper = ProposeIdentifyCropper(
        AssetBank(),
        server_dir=tmp_path / "server",
        work_dir=tmp_path / "work",
        auto_start=False,
    )
    decode_calls = 0

    def fake_decode(_segment_video: str):
        nonlocal decode_calls
        decode_calls += 1
        return frames

    monkeypatch.setattr(cropper, "_decode_frames", fake_decode)

    namer_paths = cropper.sample_namer_frames(
        "segment.mp4",
        segment_id=4,
        out_dir=tmp_path / "namer",
        count=3,
        prefix="seg00004",
    )
    crop_paths, positions = cropper.frame_paths_for_segment("segment.mp4", segment_id=4)

    assert decode_calls == 1
    assert positions == [0.2, 0.5, 0.8]
    # Preserve the namer's round(total * p) and cropper's round((total - 1) * p)
    # selection rules exactly; sharing applies only to the decode.
    assert [_pixel(path) for path in namer_paths] == [40, 100, 160]
    assert [_pixel(path) for path in crop_paths] == [40, 80, 140]


def test_decomposer_uses_optional_shared_sampler(tmp_path) -> None:
    expected = [str(tmp_path / "frame-0.png")]

    class SharedCropper:
        def sample_namer_frames(self, segment_video: str, **kwargs):
            assert segment_video == "segment.mp4"
            assert kwargs["segment_id"] == 9
            assert kwargs["count"] == 3
            assert kwargs["prefix"] == "seg00009"
            return expected

    decomposer = RoleAwareDecomposer(
        cropper=SharedCropper(),
        namer_frame_dir=tmp_path,
    )

    assert decomposer._namer_frames("segment.mp4", segment_id=9) == expected
