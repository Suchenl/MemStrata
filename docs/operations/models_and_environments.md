# MemStrata · Model weights / environments / run requirements (single authoritative list)

> Purpose: gather **all** weight locations, Python dependencies, environment
> variables, and run requirements that MemStrata uses into one place.
> When adding/swapping weights, **edit only this file**; other documents reference
> here rather than maintaining their own list.
>
> Release red line (see [`../../AGENTS.md`](../../AGENTS.md) rule 1): `src/memstrata/`
> and `src/vmem_bench/` **must not import code outside `benchmarks/VMem-Bench/`**.
> Third-party libraries and **weights loaded by path** do not count as coupling.
> The end of this file maintains a "release self-containment checklist" listing the
> cross-boundary dependencies that still remain.

## 0. Weight root (PUBLIC_MODELS_ROOT)

The code resolves local weights uniformly through the environment variable
`PUBLIC_MODELS_ROOT` (see `src/memstrata/encoders/base.py`,
`scripts/**/serve_*.sh`, and the crop server env).

| Root | Path | Purpose |
|---|---|---|
| Default root (code default) | `${PUBLIC_MODELS_ROOT}` | Default resolution root for encoder / MLLM / video weights |
| User extended root | `${PUBLIC_MODELS_ROOT}` | The full dinov3 set + **audio models** live here |

> Both roots contain a `facebook/dinov3-vitb16-pretrain-lvd1689m` of the same
> name. When running calibration / the closed loop, `export PUBLIC_MODELS_ROOT=<root>`
> for whichever root you use; anything audio-related must use the extended root
> (the default root has no `Audio/`).

## 1. Weight inventory (by role)

Path = `<PUBLIC_MODELS_ROOT>/<relative path>` unless an absolute path is noted.

### 1.1 Visual encoders (similarity gate / retrieval / scoring)
| Role | provider | Relative path | Notes |
|---|---|---|---|
| General image embed (write-path gate②/dedup/self-audit, retrieval keyframe diversity) | `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` (also `-vitl16-`/`-vits16-`, only on the extended root) | Write-path thresholds are **encoder-relative**; swapping it requires re-running calibration (see §4) |
| Text↔frame cross-modal | `siglip2` | `google/siglip2-base-patch16-512` (scoring side defaults to `-224`, `MEMSTRATA_SIGLIP2_MODEL`) | text→frame retrieval / visual scoring |
| Text↔text | `qwen3_embedding` | `Qwen/Qwen3-Embedding-4B` | Can use local weights, or set `MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT` to go through a server |
| Face (optional routing) | `insightface` | Specified by `MEMSTRATA_FACE_EMBEDDER_WEIGHTS` / `weights=` | Only when face routing is enabled |

### 1.2 Perception (crop acquisition / grounding)
| Component | Weights/dependency | Location | Notes |
|---|---|---|---|
| SAM3 concept segmentation + DINOv3 + GroundingDINO | vendored `sam3_transformers59` (transformers 5.9, cp311 .so) | `<montage_root>/models/vendor/sam3_transformers59` | **⚠️ Currently outside MemStrata** (release needs it bundled, see §6). The crop server subprocess uses a py3.11 env |
| GroundingDINO weights | Resolved together with the crop server (missing weights degrade gracefully to kind-only retrieval) | Under `PUBLIC_MODELS_ROOT` | The bbox-only fallback is tagged `bbox_high_recall_no_mask` |

### 1.3 MLLM (planner / angle attributes / scoring judgment)
Started by `scripts/memstrata/servers/serve_qwen.sh` using vLLM as an OpenAI-compatible service:
| Role | Model | Relative path | served name |
|---|---|---|---|
| Text R1/R2/R3/R5 (can also cover all roles) | Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `Qwen3.5-9B-Instruct` |
| Visual R4/R7/R8 (angle attributes / visual judgment) | Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen3-VL-8B-Instruct` |
| Judging (Stage 2 visual coverage) | qwen3-vl-32b (external judging API) | Judging service | **Calls must include the full `/chat/completions` path** (previously 404'd) |

### 1.4 Video generation / keyframes
| Component | Weights | Location | Notes |
|---|---|---|---|
| Video i2v (default backend) | Wan2.2-I2V-A14B lightx2v 4-step distill | `${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step` | Assembled by `setup_lightx2v_weights.sh` from two distilled safetensors + the base model; config at `configs/video_gen/wan22_i2v_a14b_lightx2v_4step.toml` |
| Keyframe fusion | FLUX.2 klein (`flux.2-klein-9b-kv-fp8`) | See the corresponding image backend config | R3/R4 collage → FLUX I2I, can be turned off with `--no-flux` |

### 1.5 Audio (showcase / audio MVP, weights under the extended root `Audio/`)
Root: `${PUBLIC_MODELS_ROOT}/Audio`
| Purpose | Model | Subpath |
|---|---|---|
| Dialogue TTS | CosyVoice2-0.5B | `FunAudioLLM/CosyVoice2-0.5B` |
| Dialogue TTS (alternative/new version) | Fun-CosyVoice3-0.5B-2512 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| CosyVoice text frontend | CosyVoice-ttsfrd | `FunAudioLLM/CosyVoice-ttsfrd` |
| Emotional/bilingual TTS (alternative) | IndexTTS-2 | `IndexTeam/IndexTTS-2` |
| Background music (BGM) | ACE-Step v1.5 | `ACE-Step/acestep-v15-sft`, `ACE-Step/acestep-v15-xl-turbo-diffusers` |
| Sound effects (foley) | Stable Audio Open | `stabilityai/stable-audio-open-1.0`, `stabilityai/stable-audio-open-small` |

> For the design, see [`../showcase/audio_pipeline_PLAN.md`](../showcase/audio_pipeline_PLAN.md) (Chinese)
> (CosyVoice2 dialogue / ACE-Step BGM / Stable Audio Open foley + ducking +
> alignment). The weights are downloaded; the MVP is pending wiring.

## 2. Python environments

Use a CPython 3.10+ interpreter with the packages you actually run. There is no
required conda env name.

| Role | Typical stack |
|---|---|
| CPU tests / `recording` demo | `pip install -e ".[dev]"` (numpy, pillow, pytest) |
| Perception / SAM3 crop server | CPython **3.11** + torch + vendored `models/vendor/sam3_transformers59` (transformers 5.9). Set `MEMSTRATA_PYTHON` if that is not your default `python3`. |
| Wan / LightX2V generation | torch + flash-attn matching the chosen backend (see the backend TOML) |
| MLLM judge / IAMFlow HTTP | vLLM or any OpenAI-compatible server; point `*_ENDPOINT` env vars at it |

## 3. Environment variables (MEMSTRATA_*)

| Variable | Effect | Default |
|---|---|---|
| `PUBLIC_MODELS_ROOT` | Local weight resolution root | `${PUBLIC_MODELS_ROOT}` |
| `MEMSTRATA_GENERAL_EMBEDDER_PROVIDER` | General image embed provider (`hash`\|`dinov3`\|...) | `hash` (non-semantic, gate②/self-audit off) |
| `MEMSTRATA_FACE_EMBEDDER_PROVIDER` / `MEMSTRATA_PLACE_EMBEDDER_PROVIDER` | Face / place routing provider | empty (no routing) |
| `MEMSTRATA_SIGLIP2_WEIGHTS` / `MEMSTRATA_SIGLIP2_MODEL` | siglip2 weights/model-name override | relative default |
| `MEMSTRATA_QWEN3_EMBEDDING_ENDPOINT` / `_MODEL` / `_API_KEY` / `_WEIGHTS` | qwen3 text embed via server or local | empty → local weights |
| `MEMSTRATA_ANGLE_CLASSIFIER` | Angle/attribute classifier `null`\|`heuristic`\|`vlm` | `null` (per-rep angle unknown, **stratified results must not be reported**) |
| `MEMSTRATA_CONTEXT_JUDGER_BASE_URL` / `MEMSTRATA_CROP_ATTR_BASE_URL` / `..._MODEL` | planner / angle-attribute MLLM endpoint and model | see `mllm/planner.py`, `mllm/crop_attributes.py` |
| `MEMSTRATA_SCORING_EMBEDDER_WEIGHTS` / `MEMSTRATA_WEIGHTS_ROOT` | Scoring-side encoder weights | see `vmem_bench/scoring/embedder.py` |
| `MEMSTRATA_ALLOW_HF_DOWNLOAD` | =1 allows transformers online resolution (offline by default) | unset (offline; the crop server sets `HF_HUB_OFFLINE=1`) |
| `MEMSTRATA_RETRIEVAL_VARIANT` / `MEMSTRATA_RETRIEVAL_TOPK` | Retrieval baseline family variant / top-k | see `retrieval_family_DESIGN.md` |

## 4. Write-path threshold calibration (must be run when swapping encoders)

The per-type thresholds in `MemoryPolicy.production()` (B_τ / γ_τ / β_τ / cohesion
floor) are **encoder-relative** starting values, not measured values. After
swapping the encoder (e.g. hash→dinov3) you must re-run:

```bash
PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} \
python3 \
  experiments/methods/MemStrata/20260725_memstrata_write_path_calibration/calibrate_write_path.py \
  --labels experiments/methods/MemStrata/20260722_memstrata_cohesion_calibration/labels_lsmdc.json \
  --encoder dinov3
```

- Labeled reference set: `experiments/methods/MemStrata/20260722_memstrata_cohesion_calibration/labels_lsmdc.json`
- dinov3 cached embeddings already exist in the 20260722 directory (`embeddings_dinov3.json`, reusable and GPU-free)
- GPU-free pipeline self-test: `--self-test`
- The `suggested_policy` in the produced `calibration_result.json` is backfilled into `MemoryPolicy.production()`

## 5. Run requirements (GPU / remote)

- GPU priority: large-VRAM cards first; **only schedule on nodes allocated to you**.
- Remote jobs must use `setsid` (or equivalent) to survive SSH/tmux disconnects.
- The judging API must include the full `/chat/completions` path.

## 6. Release self-containment checklist (release blockers)

Before release, the following "outside MemStrata" dependencies must be eliminated
(`src/memstrata/` must not import other repo paths):

- [ ] `models/vendor/sam3_transformers59` (crop server dependency) is outside `benchmarks/VMem-Bench/` — needs to be bundled or explicitly declared as an optional external dependency.
- [ ] `src/memstrata/skills/embedding_deduplication/` (tests + README) `import montage.skills.embedding_deduplication` — needs to be vendored or removed.
- [ ] `src/memstrata/skills/focus_segmentation/` (tests + README) `import montage.skills.focus_segmentation` — same as above.
- [ ] `src/memstrata/skills/layout_anchor_processing/` is marked "vendored from montage" — confirm it is genuinely vendored with no runtime cross-references.
- [ ] The video lightx2v weight directory is a to-be-assembled TODO: `.../Wan-AI/Wan2.2-I2V-A14B-lightx2v-4step` (user is downloading).

> Verification command (should be empty): under `src/memstrata/` and
> `src/vmem_bench/`, search for `import montage` / `from montage` / any output
> writes that point to other repo paths.
