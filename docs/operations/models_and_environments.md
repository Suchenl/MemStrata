# MemStrata · Model weights / environments / run requirements (single authoritative list)

> Purpose: gather **all** weight locations, Python dependencies, environment
> variables, and run requirements that MemStrata uses into one place.
> When adding/swapping weights, **edit only this file**; other documents reference
> here rather than maintaining their own list.
>
> Release red line (see [`../../AGENTS.md`](../../AGENTS.md) rule 1): `src/memstrata/`
> must not import code from another repository. The benchmark package is a
> separate project; integration code belongs in VMem-Bench adapters.
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

## 4. Write-path threshold calibration

The per-type thresholds in `MemoryPolicy.production()` are
**encoder-relative starting values**, not universal constants. The release
does not ship private calibration movies or labels. If you replace the image
encoder, calibrate on a licensed validation set from your own application,
record the encoder and thresholds in your experiment log, and do not compare
those numbers directly with the paper's frozen run.

## 5. Run requirements (GPU / remote)

- GPU priority: large-VRAM cards first; **only schedule on nodes allocated to you**.
- Remote jobs must use `setsid` (or equivalent) to survive SSH/tmux disconnects.
- The judging API must include the full `/chat/completions` path.

## 6. External runtime services

These are runtime integrations, not Python imports:

- **WeDetect-Ref** is the faithful default crop grounder. Run the separately
  licensed service and set `MEMSTRATA_WEDETECT_URL`; when it is unavailable,
  MemStrata can use the optional SAM3 fallback.
- The SAM3 fallback needs the separately supplied
  `transformers>=5.9` compatibility bundle and `MEMSTRATA_SAM3_DEPS`.
- FLUX, Wan/LightX2V, and Qwen weights are never bundled. Put them under
  `PUBLIC_MODELS_ROOT` exactly as listed in `MODELS.md`.

The source package itself is standalone: it does not import VMem-Bench or any
other project package. The adapter that evaluates MemStrata lives in the
VMem-Bench repository.
