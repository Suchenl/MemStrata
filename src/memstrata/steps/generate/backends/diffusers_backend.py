"""Real reference-conditioned continuation backend (diffusers families).

This is the production counterpart of :class:`OracleMediaGenerator`: it implements the same
``MediaGenerationBackend`` contract (``generate(task) -> GenerationArtifact``) but synthesizes
the segment with a real text/image-to-video model instead of slicing the source. It is the only
component that differs between Oracle Replay and a real production run (README sec. 10/11).

Supported families (selected by ``models/model_configs/video_gen/<name>.toml``; the toml's
``family`` picks the diffusers pipeline classes, ``model`` picks the weights -- so e.g. VACE-1.3B
and VACE-14B are two configs of the same ``wan_vace`` family):

* ``wan_vace`` -- Wan2.1-VACE reference-conditioned generation (the project's primary backend;
  consumes Composed-Context reference images as ``reference_images``; 1.3B and 14B).
* ``wan_t2v``  -- Wan2.1 text/image-to-video (1.3B and 14B).
* ``ltx``      -- LTX-Video text/image-to-video (fast "frozen smoke" backend).
* ``cogvideox``-- CogVideoX text/image-to-video (2B and 5B).
* ``hunyuan``  -- HunyuanVideo text/image-to-video.
* ``mochi``    -- Mochi-1 text-to-video (no reference conditioning).

Each family entry declares its text-to-video class (``t2v_cls``), optional image/reference class
(``i2v_cls``), and the keyword its reference argument uses (``ref_arg``: ``reference_images`` |
``image`` | ``None``). Adding another diffusers video family is a data-only change here plus a
matching config toml -- no new code path.

Design constraints honored here:

* Heavy deps (``torch``/``diffusers``/``Pillow``) are imported lazily inside ``generate`` so the
  module imports cheaply on any host (tests, CI, macOS) and only the chosen real run pays for them.
* The backend conditions on the *Composed Context*: the production loop resolves the segment's
  selected assets to still-image reference paths and passes them on ``task.controls['composed_references']``.
  Composition stays model-free; this adapter only materializes what the plan already selected.
* On a host without the model deps / weights / a usable accelerator, ``generate`` raises a clear
  ``RuntimeError`` (callers such as ``scripts/create.py`` catch it and fall back to the frozen
  Oracle backend), mirroring how the segmentation/embedding stages degrade gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 VACE env.
    import tomli as tomllib

from memstrata.steps.generate.backends._support import ArtifactStore as ProjectService
from memstrata.steps.generate.backends._support import RunContext as ProjectContext, new_id
from memstrata.steps.generate.schemas import (
    GenerationArtifact,
    MediaGenerationTask,
    MediaTaskType,
)

# family -> diffusers pipeline classes + reference arg + recommended params. The toml config
# overrides ``model`` and any numeric param. ``t2v_cls``/``i2v_cls`` are diffusers class names;
# ``ref_arg`` is how the family ingests Composed-Context reference images (None = text-only).
_FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "wan_vace": {
        "model": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "t2v_cls": "WanVACEPipeline", "i2v_cls": "WanVACEPipeline", "ref_arg": "reference_images",
        "num_frames": 49, "fps": 16, "height": 480, "width": 832, "steps": 30, "guidance_scale": 5.0,
    },
    "wan_t2v": {
        "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "t2v_cls": "WanPipeline", "i2v_cls": "WanImageToVideoPipeline", "ref_arg": "image",
        "num_frames": 49, "fps": 16, "height": 480, "width": 832, "steps": 30, "guidance_scale": 5.0,
    },
    "ltx": {
        "model": "Lightricks/LTX-Video",
        "t2v_cls": "LTXPipeline", "i2v_cls": "LTXImageToVideoPipeline", "ref_arg": "image",
        "num_frames": 97, "fps": 24, "height": 480, "width": 704, "steps": 40, "guidance_scale": 3.0,
    },
    "cogvideox": {
        "model": "THUDM/CogVideoX-2b",
        "t2v_cls": "CogVideoXPipeline", "i2v_cls": "CogVideoXImageToVideoPipeline", "ref_arg": "image",
        "num_frames": 49, "fps": 8, "height": 480, "width": 720, "steps": 50, "guidance_scale": 6.0,
    },
    "hunyuan": {
        "model": "hunyuanvideo-community/HunyuanVideo",
        "t2v_cls": "HunyuanVideoPipeline", "i2v_cls": "HunyuanVideoImageToVideoPipeline", "ref_arg": "image",
        "num_frames": 61, "fps": 15, "height": 480, "width": 832, "steps": 30, "guidance_scale": 6.0,
    },
    "mochi": {
        "model": "genmo/mochi-1-preview",
        "t2v_cls": "MochiPipeline", "i2v_cls": None, "ref_arg": None,
        "num_frames": 61, "fps": 30, "height": 480, "width": 848, "steps": 64, "guidance_scale": 4.5,
    },
}


class DiffusersVideoBackend:
    """Reference-conditioned continuation generator over a diffusers video pipeline.

    Construct via :meth:`from_config` for a real run, or directly with an injected
    ``pipeline_factory`` / ``exporter`` for testing without the heavy deps.
    """

    def __init__(
        self,
        context: ProjectContext,
        *,
        family: str,
        params: dict[str, Any],
        run_id: str,
        project_service: ProjectService | None = None,
        pipeline_factory: Callable[[], Any] | None = None,
        exporter: Callable[[list, str, int], None] | None = None,
    ) -> None:
        self.context = context
        self.family = family
        # Family defaults supply the pipeline classes + reference arg; explicit params override.
        self.params = {**_FAMILY_DEFAULTS.get(family, {}), **params}
        self.run_id = run_id
        self.project_service = project_service or ProjectService()
        self._pipeline_factory = pipeline_factory
        self._exporter = exporter
        self._pipelines: dict[bool, Any] = {}
        from memstrata.steps.generate.backends._support import resolve_model_reference

        self.model_name = resolve_model_reference(str(params.get("model", family)))

    @classmethod
    def from_config(
        cls,
        name: str,
        context: ProjectContext,
        project_service: ProjectService,
        run_id: str,
        models_config: Path,
    ) -> "DiffusersVideoBackend":
        config = _load_video_gen_config(name, models_config)
        family = str(config.get("family", name)).lower()
        defaults = _FAMILY_DEFAULTS.get(family)
        if defaults is None:
            raise ValueError(
                f"Unknown video-gen family '{family}' for backend '{name}'. "
                f"Known families: {sorted(_FAMILY_DEFAULTS)}"
            )
        params = {**defaults, **{k: v for k, v in config.items() if k != "family"}}
        return cls(
            context, family=family, params=params, run_id=run_id, project_service=project_service
        )

    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        if task.task_type is not MediaTaskType.VIDEO_SEGMENT:
            raise ValueError("DiffusersVideoBackend only supports video_segment tasks")

        notes: list[str] = []
        references = _reference_image_paths(task)
        can_condition = bool(references) and self.params.get("ref_arg") is not None
        if references and not can_condition:
            notes.append(f"family={self.family}_is_text_only:references_ignored")
        fps = int(self.params.get("fps", 16))
        num_frames = _resolve_num_frames(task, self.params, fps)
        pipeline = self._ensure_pipeline(can_condition)
        frames = self._run_pipeline(
            pipeline, task, references if can_condition else [], num_frames, notes
        )

        out_dir = self.context.workspace_path / "runs" / self.run_id / "gen_tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task.segment_id}.mp4"
        self._export(frames, str(out_path), fps)

        digest, object_path = self.project_service.import_object(self.context, out_path)
        out_path.unlink(missing_ok=True)
        if references:
            notes.append(f"conditioned_on={len(references)}_reference_images")
        return GenerationArtifact(
            artifact_id=new_id("artifact"), task_id=task.task_id, segment_id=task.segment_id,
            media_type="video", object_hash=digest, object_uri=str(object_path),
            model_name=self.model_name, degradation_notes=notes,
        )

    def _ensure_pipeline(self, conditioned: bool) -> Any:
        if conditioned not in self._pipelines:
            if self._pipeline_factory is not None:
                self._pipelines[conditioned] = self._pipeline_factory()
            else:
                self._pipelines[conditioned] = self._default_pipeline_factory(conditioned)
        return self._pipelines[conditioned]

    def _default_pipeline_factory(self, conditioned: bool) -> Any:
        try:
            import torch  # noqa: PLC0415
            import diffusers  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only on a configured host
            raise RuntimeError(
                "Real video generation needs the 'generation' extra (torch + diffusers). "
                "Install it (scripts/install.sh) or use the frozen Oracle backend."
            ) from exc

        from memstrata.steps.generate.backends._support import hf_cache_dir  # noqa: PLC0415

        device, dtype = _select_device(torch)
        if device == "cpu":  # pragma: no cover - guarded run path
            raise RuntimeError(
                f"No CUDA/MPS accelerator available for family '{self.family}'. "
                "Run on a GPU host or use the frozen Oracle backend."
            )
        pipeline_cls = self._pipeline_class(diffusers, conditioned)
        # Weights go to / load from models/model_weights (project model-weights rule).
        pipe = pipeline_cls.from_pretrained(
            self.model_name, torch_dtype=dtype, cache_dir=hf_cache_dir()
        )
        pipe = pipe.to(device)
        return pipe

    def _pipeline_class(self, diffusers: Any, conditioned: bool) -> Any:
        key = "i2v_cls" if conditioned else "t2v_cls"
        cls_name = self.params.get(key) or self.params.get("t2v_cls")
        if not cls_name:  # pragma: no cover - guarded by config validation
            raise ValueError(f"family '{self.family}' has no pipeline class for conditioned={conditioned}")
        return getattr(diffusers, cls_name)

    def _run_pipeline(
        self, pipeline: Any, task: MediaGenerationTask, references: list[Path],
        num_frames: int, notes: list[str],
    ) -> list:
        kwargs: dict[str, Any] = {
            "prompt": task.prompt,
            "num_frames": num_frames,
            "height": int(self.params.get("height", 480)),
            "width": int(self.params.get("width", 832)),
            "num_inference_steps": int(self.params.get("steps", 30)),
            "guidance_scale": float(self.params.get("guidance_scale", 5.0)),
        }
        negative = task.controls.get("negative_prompt")
        if negative:
            kwargs["negative_prompt"] = str(negative)
        if references:
            kwargs.update(self._reference_kwargs(references, notes))
        result = pipeline(**kwargs)
        return result.frames[0]

    def _reference_kwargs(self, references: list[Path], notes: list[str]) -> dict[str, Any]:
        """Map Composed-Context reference images onto the family's conditioning argument."""

        ref_arg = self.params.get("ref_arg")
        if not ref_arg:  # pragma: no cover - guarded by can_condition in generate()
            return {}
        try:
            from memstrata.lib.media import load_crop_rgb_for_model  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reference conditioning needs Pillow (the 'vision' extra).") from exc
        images = [load_crop_rgb_for_model(p) for p in references]
        if ref_arg == "reference_images":
            return {"reference_images": images}
        notes.append("reference_anchor=first_image")
        return {ref_arg: images[0]}

    def _export(self, frames: list, out_path: str, fps: int) -> None:
        if self._exporter is not None:
            self._exporter(frames, out_path, fps)
            return
        try:
            from diffusers.utils import export_to_video  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Exporting video needs diffusers (the 'generation' extra).") from exc
        export_to_video(frames, out_path, fps=fps)


def _load_video_gen_config(name: str, models_config: Path) -> dict[str, Any]:
    path = Path(models_config) / "video_gen" / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No video-gen config for backend '{name}' at {path}. "
            f"Add a TOML under {Path(models_config) / 'video_gen'}."
        )
    data = tomllib.loads(path.read_text())
    models = data.get("models", {})
    if not models:
        raise ValueError(f"Video-gen config {path} has no [models.*] table")
    # Single backend per file: take the first (and typically only) model table.
    return next(iter(models.values()))


def _reference_image_paths(task: MediaGenerationTask) -> list[Path]:
    refs = task.controls.get("composed_references") or []
    paths: list[Path] = []
    for ref in refs:
        uri = ref.get("image") if isinstance(ref, dict) else ref
        if not uri:
            continue
        path = Path(str(uri))
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            paths.append(path)
    return paths


def _resolve_num_frames(task: MediaGenerationTask, params: dict[str, Any], fps: int) -> int:
    duration = task.controls.get("duration_sec")
    if isinstance(duration, (int, float)) and duration > 0:
        return max(1, int(round(float(duration) * fps)))
    return int(params.get("num_frames", 49))


def _select_device(torch: Any) -> tuple[str, Any]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32
