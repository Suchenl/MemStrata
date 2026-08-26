# MemStrata

# Getting started (no GPU)

Clone the two repos **next to each other**:

```bash
git clone https://github.com/Suchenl/MemStrata.git
git clone https://github.com/Suchenl/VMem-Bench.git
cd MemStrata
python -m pip install -e ".[dev]"
python scripts/memstrata/doctor.py
bash scripts/memstrata/cpu_demo.sh
```

That writes a stitchable mp4 + `bank.json` under `production/outputs/`. Weights, Qwen, and Wan are not used.

GPU / paper tables: [`MODELS.md`](MODELS.md) and [`REPRODUCE.md`](REPRODUCE.md). Benchmark: [VMem-Bench](https://github.com/Suchenl/VMem-Bench). Gold: [huggingface.co/datasets/Suchenl/VMem-Bench](https://huggingface.co/datasets/Suchenl/VMem-Bench).

The generator \(G_\theta\) is **swappable**. Default production path (documented, not hard-wired): FLUX.2 Klein 9B-KV keyframes → Wan2.2-I2V-A14B LightX2V 4-step. List backends with `python -m memstrata.production.run --list-backends`.

```bash
python -m pytest -q
```

`memstrata` and `vmem_bench` never import each other. Evaluation adapters live only in the VMem-Bench repo.

## Citation

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

See [`CITATION.cff`](CITATION.cff). Code is Apache-2.0.

---

# MemStrata（方法说明）

**面向可控长视频生成的记忆管理与上下文组合方法包。** 把一部长视频里出现过的实体
（`character` / `prop` / `location`）沉淀为**结构化、可被生成条件化的记忆库** \(\mathcal{M}_n\)，
并在每个生成步为"点名到的实体"组合出**最小而充分**的视觉上下文，让下游生成器稳定复现同一身份。

> 记忆库不是"帧的仓库"，而是"**每个实体一份、可被生成条件化的身份档案**"。

本包（Python 包名 `memstrata`）与评测基准 [VMem-Bench](https://github.com/Suchenl/VMem-Bench)
（包名 `vmem_bench`）**零相互导入、互不外部引用**（自包含硬约束见 [`AGENTS.md`](AGENTS.md)）。
评测只在 VMem-Bench 仓库的 `scripts/evaluate_baselines/` adapter 里把本方法当黑盒 import。

**最高纲领**是 [`src/memstrata/docs/design_philosophy.md`](src/memstrata/docs/design_philosophy.md)
（六条库质量公理 + WHO-before-WHERE 准入原则）；任何记忆相关取舍先服从它，本文件与之冲突时以纲领为准。

---

## 目录布局（方法侧，post-split）

```
.
├── AGENTS.md
├── README.md
├── configs/
├── docs/
├── src/memstrata/
│   ├── docs/design_philosophy.md
│   ├── bank/
│   ├── skills/
│   ├── mllm/
│   ├── steps/
│   ├── pipeline.py
│   ├── production/run.py
│   ├── adapters/
│   ├── encoders/
│   └── lib/
├── production/               # screenplay samples (run outputs gitignored)
└── scripts/memstrata/
```

---

## 四阶段因果记忆环（对齐论文 §4）

$$(q_n,\tilde p_n,\mathcal{C}_n)=\mathcal{R}(g_n,\mathcal{M}_n)\ \to\ x_n=G_\theta(\tilde p_n,\Phi(\mathcal{C}_n))\ \to\ \mathcal{O}_n=\mathcal{A}(x_n,q_n)\ \to\ \mathcal{M}_{n+1}=\mathcal{W}(\mathcal{M}_n,\mathcal{O}_n)$$

| 论文阶段 | 中文 | 承载技能（`skills/…`） | 一句话 |
|---|---|---|---|
| Intent Interpretation | **意图解读** | `intent_understanding` | 外部意图 \(g_n\)→结构化请求 \(q_n\)：选哪些已存 id + 角度/状态偏好；FAST 名字匹配（0 模型），SLOW 才用 MLLM |
| （读路径解引用） | 上下文组合 | `composition` + `memory_retrieval` | 把 \(q_n\) 里的 id 确定性解引用为 \(\mathcal{C}_n\)（LexMax：显式→状态→视角→最新）；`memory_retrieval.name_match` 是 FAST 锚 |
| Visual Generation | 视觉生成 | `generation_routing` + `steps/generate` | 参考条件化 backend 生成 \(x_n\)；本方法不改 \(G_\theta\) |
| Evidence Acquisition | **证据获取** | `decomposition` + `crop_acquisition` | \(\mathcal{O}_n=\mathcal{O}^{\mathrm{req}}\cup\mathcal{O}^{\mathrm{disc}}\)：点名实体定向取证 + 类型受限自我发现 |
| Stratified Update | **记忆更新**（原“落库/curation”） | `memory_update`（原 `memory_curation`；类 `MemoryUpdater`，原 `AssetCurator`） | 身份锚定/重认 → 新颖度 → 冲突淘汰 → 预算 B_τ → 回溯自审 → **导出记忆快照 JSON** |

其余技能：`optimization`（读侧监控/调参）、`layout_anchor_processing`、`embedding_deduplication`、
`focus_segmentation`（可复用算法件）。

---

## ★ 系统三大核心

这三步是整个系统的核心，其余都是围绕它们的工程支撑。

> **贯穿三核心的总原则：快慢思考结合，缺一不可。** 能确定性/名字匹配直接解决的走**快思考**（0 模型调用，
> 关乎生成反应速度）；只有需要语义判断的才升级到**慢思考**（MLLM）。举例：意图里"点名已存实体"是快；
> "为一个新状态判断哪些旧状态值得参考"（人老了但年轻的脸可参考）是慢；"把检测到的图匹配到新实体描述"是慢；
> "自我发现帧里值得记的实体"是慢。**快慢不是两条路线，而是同一条流水线上按需升级的两档。**

### 核心 1 · 意图解读（Intent Interpretation）

把当前意图 \(g_n\) 解析为 typed 请求 \(q_n=\{(e_i,\kappa_i,c_i)\}\)：选中哪些**已可寻址**的资产 id、
以什么角度/状态/连续性/排除约束。**FAST**（默认，0 模型调用）走名字/别名匹配
（复用 `memory_retrieval.name_match`）；只有名字缺失或歧义时才降级到 **SLOW**（`MllmIntentResolver`
在 bank 清单上推理，且绝不臆造 bank 之外的 id）。

> **关键：证据获取的方式（快 or 慢、取哪些参考图），是在这一步就规划好的。** 见核心 2 的取证路径表。

### 核心 2 · 证据获取（Evidence Acquisition）

> 以下为用户原话，作为本系统证据获取的权威规则，原封不动记录：
>
> 实体获取的时候，其实分为两部分，一部分是提示词指代，这部分的实体获取通过 1: 实体匹配，是否匹配上实体名称，如果匹配上，那就看这个实体有没有指定新的外观状态，如果没有那就从记忆中拿图。如果有的话，那就看看有什么新状态，拿一些可以被参考的图组成上下文。例如人老了但是年轻的时候的人脸还是可以参考一下的。 然后这些新状态生成的结果会被记录下来。 2. 如果没匹配上实体，但又有名字和描述，那就说明新生成的这个要被存储为新实体。生成之后做检测，然后再用VLM去做匹配，看看检测出来的所有图里哪些图是符合新实体的描述，然后存储实体图。这些都是属于慢思考，是从意图解读那一步就要规划好的。
>
> 还有一部分是自我获取，即模型生成了一些不在提示词中的内容，但是又值得被记忆下来。

> **说明（对上述原话的精化）：** 这些路径**不是全部慢思考，而是快慢结合**——能名字匹配直接解决的是快
> （0 模型调用），只有需要语义判断的分支（新状态取哪些旧状态参考、把检测图匹配到新实体描述）才升级到慢。

对应论文 \(\mathcal{O}_n=\mathcal{O}^{\mathrm{req}}\cup\mathcal{O}^{\mathrm{disc}}\)，落成四条可执行路径：

| 路径 | 快/慢 | 触发（意图解读判定） | 生成前做什么 | 生成后做什么（取证+记忆） |
|---|---|---|---|---|
| **A. 提示词指代·已匹配·无新状态** | **快**（0 模型） | 名字命中已存实体，且未指定新外观 | 直接从记忆取该实体的参考图组上下文 | 一般无需新增取证（可选：出现真正新视角/状态才补记一条） |
| **B. 提示词指代·已匹配·有新状态** | **慢** | 名字命中，但指定了新外观状态（如"变老"） | **MLLM 判断哪些旧状态值得参考**（如年轻时人脸）组成条件上下文 | 对生成结果做定向取证，抠出**新状态** crop，记为新 state 落库 |
| **C. 提示词指代·未匹配但有名字+描述** | **慢** | 名字未命中，但意图给了名字+描述 → 视为**新实体（provisional id）** | 无参考图（首见） | 生成后**确定性检测**候选 → **VLM 批量匹配**"哪些检测图符合该新实体的名字+描述" → 存中选的实体图 |
| **D. 自我获取（discovered）** | **慢** | 生成里出现了不在提示词中的实体 | — | 类型受限自我发现（VlmEntityDecomposer 提命名候选）→ 去掉被 A/B/C 覆盖的 → 走身份重认落库 |

> A/B/C 属于 \(\mathcal{O}^{\mathrm{req}}\)（点名取证，身份由命名确定性锚定）；D 属于 \(\mathcal{O}^{\mathrm{disc}}\)
> （无符号 id，身份由类型受限重认 χ 决定）。**命名永不由 proposer 臆造**——身份始终是写路径可复现的决策。

### 核心 3 · 记忆更新（Memory Update，原 memory_curation / `AssetCurator`→`MemoryUpdater`）

把证据获取产出的 `Observation` 更新进分层 `AssetBank`：身份锚定/重认（χ）→ 兼容分层内的新颖度去重
→ 状态冲突淘汰（留痕）→ per-type 预算 B_τ 的属性多样性选择 → 每 chunk 回溯 cohesion 自审隔离内鬼。
遵循 WHO-before-WHERE：一条证据先证明"是本实体且看得清"，才有资格参与角度/状态多样性竞争。

#### 记忆快照（对外产物）：动态更新的 JSON + 同根 `visual/` 图库

记忆更新的**对外产物**不再是杂乱的 rep-centric dump，而是一份**人类可读、随每步动态更新**的记忆快照
JSON（结构与基准 gt 的 `entities` 同源），并把视觉记忆按同根目录的 `visual/` 分层存放：

```
<run_dir>/
├── long_video.mp4                     # 成片：每生成一段就拼接到尾部（记忆的时间轴基准）
├── memory.json                        # 记忆快照（每步更新）
└── visual/
    ├── characters/<asset_id>/states/<state>/*.png
    ├── props/<asset_id>/states/<state>/*.png
    └── locations/<asset_id>/states/<state>/*.png
```

`memory.json` schema（`memstrata-memory-1.0`）：

```json
{
  "schema": "memstrata-memory-1.0",
  "movie_id": "...", "fps": 24.0, "updated_sec": 1234.5,
  "video": {"path": "long_video.mp4", "duration_sec": 1234.5},
  "entities": {
    "character_elias": {
      "name": "伊莱亚斯", "kind": "character",
      "description": "灰白胡须的守塔人…",
      "aliases": ["守塔人"], "lifecycle": "reusable",
      "initial_state": "default", "first_seen_sec": 12.0,
      "states": {
        "default": {
          "description": "中年硬朗…",
          "first_seen_sec": 12.0,
          "appearances": [{"sec": 12.0, "chunk": 0}, {"sec": 88.5, "chunk": 5}],
          "images": ["visual/characters/character_elias/states/default/c00000_front.png"]
        },
        "changed": { "description": "拄杖迟暮…", "first_seen_sec": 640.0, "appearances": [...], "images": [...] }
      }
    }
  }
}
```

要点：
- **同级成片 `long_video.mp4`**：每生成一段就拼到尾部；`memory.json` 里所有 `sec`/`first_seen_sec` 都以**这条成片的时间轴**为准，`video.duration_sec` 记录当前总时长。
- **实体 → 状态 → 视觉记忆**三级；每个状态各自挂**相对路径**图片（同根 `visual/…/states/<state>/`）。
- **时间记忆**：实体级 `first_seen_sec` + 每个状态的 `first_seen_sec` 与 `appearances[{sec, chunk}]`（初次 + 每次出现），**尽量精细到秒**（秒不可得时退化记 chunk）。
- 状态键来自 `state_angle`（default/changed/damaged）或意图指定的具名新状态；描述取该状态的观测/意图描述。
- 由 `memory_update` 的 exporter 在每次 chunk 落库后原子写出（write-temp→rename），是"动态更新的 json"。

---

## ★ 写侧增强计划（已批准 / 实现中）

针对当前写侧质量门控的缺口，给出一个**效率高、效果好、且不平白增加 VLM 调用**的方案。

### 设计原则

1. **确定性优先、免费门控先行**：暗/过曝/碎片/尺寸/清晰度这类判定全部确定性完成，**0 模型调用**；
   只有通过确定性预筛的 crop 才有资格进入 VLM。
2. **读写侧 VLM 预算不对称**（关乎生成反应速度 vs 记忆质量）：
   - **读侧（意图解读/组合）：尽量只调 1 次**，甚至 0 次——读侧直接影响生成响应速度。名字匹配全走快思考；
     只有名字缺失/歧义、或需要"为新状态挑旧参考"时才升级 1 次慢思考。
   - **写侧（记忆更新）：可以多调，但要 lean**（不为多而多）——写侧不在生成关键路径上，多花一点保证记忆库质量。
     仍以"每 chunk 批量合并"为纪律：一次"帧级发现+命名"，一次"crop 级批量属性+新实体验证"，**杜绝 per-crop / per-gate 碎调用**。
     唯一按需的第三类写侧调用是**身份仲裁**（VLM-first identity gate）：只对*自我发现*（discovered）观测、且落在编码器分 χ *灰区*时才触发，一次调用把"进来的 crop + 候选的 top-k 跨分层参考图"一起送进去判（多参考、单调用）；
     编码器短路/清晰拒/强模糊 defer 先把绝大多数挡在 VLM 之前，请求侧（A/B/C 点名取证）永不触发。
   - **命名：一次吐全部 + 对"没命中桶"消歧**——`VlmEntityDecomposer.propose` **一次调用**吐出所有实体:label 在提示词里的即**命中**(点名/requested),其余进"没命中桶"。
     但没命中桶里混了两类**字符串上分不开**的东西:**取错名**(提示词里的实体被 VLM 翻译/改写跑偏,如 `紫色小鸟`→`purple bird`)与**真自发现**(提示词根本没提的实体)。
     确定性 `term_in_prompt`(0 调用)先把已命中的择出去;剩下的没命中标签才批量送进**一次全新上下文的审核调用**(另起一轮、带帧,绝非自吐自审),逐个判:
     `in_prompt=true`→取错名,掰回提示词原话(**转为命中**);`in_prompt=false`→真自发现,保留描述名。掰回的名再用 `term_in_prompt` **确定性复核**,查不到即判幻觉、退回原标签。
     整段是"命中即免费、没命中才消歧、改名必复核",不为多而多。
   - **抠图概念由 call1 感知驱动(修 prop 召回)**——SAM3-concept 是开放词表,给具体名词("red apple"/"acorn")才抠得出,给死板的 `"object"` 基本零候选(实测 4 个 prop 帧过 QA 的候选:`object`=0,具体名词=8)。
     故 call1 每个实体多返一个**短英文 `category`**,写路径用它当 SAM3 概念、原来的 kind 概念(person/animal、object)只做兜底并集;character 因为 person/animal 本就够用不受影响。此外动物一律判 character(蝴蝶不再误入 prop)。
3. **一次调用多返字段**：状态/视角/景别/光照/遮挡/`identity_visible`/是否符合新实体描述，全部在同一个
   结构化 VLM 请求里一次返回（纲领 §2：③ identity_visible 就是"复用已有 crop 属性调用多返一个布尔"）。

### 缺口对照（现状 → 提议）

| # | 你点名的能力 | 现状 | 提议 |
|---|---|---|---|
| 1 | **VLM 再审：分割对不对** | ❌ 首见无任何身份门控（exemplar 空→直接收 SAM3 最高分），易让错主体成 seed 污染后续 | 首见 seed 折进那**一次批量 crop 调用**：问"这是不是一个符合 `<type>`/`<desc>` 的实体"，不符合则拒（0 额外调用） |
| 1 | **VLM 再审：质量好不好** | 质量为确定性 QA（几何/清晰度/暗），非 VLM | 保持确定性（更便宜更稳）；VLM 只判语义符合度，不判像素质量 |
| 1 | **提取状态+描述记录** | `VlmCropAttributeClassifier` 有能力但默认 `null`；bench mllm adapter 用 `NullCropAttributeClassifier`→未提取 | 把状态/描述/视角/景别/光照/遮挡/identity_visible 合进那**一次批量 crop 调用**，写进 rep.annotations |
| 2 | **连通域检测（防碎图）** | ✅ 已有 `mask_quality.assess_mask_quality`（最大 CC≥0.90、显著 CC≤1、空洞≤0.12） | 保留；补：GDINO bbox-only 兜底无掩膜，标 `needs_review` 时也给一条尺寸/长宽比确定性护栏 |
| 3 | **亮度检测（防太亮/太暗）** | ⚠️ 只有"太暗"（`is_dark_low_information`），**无过曝** | 在 `lib/crop_quality.py` 补对称的 `is_overexposed_low_information`（均值>235 且 std<12），接入 `crop_qa.audit_crop` 与记忆更新 gate ① |
| 4 | **门控用状态新颖度（非仅视觉）** | ⚠️ 结构已有（`compatible_stratum` 含 `state_angle`），但默认 state 全 `unknown`→退化为纯视觉 | ①靠上面的批量调用把 `state_angle` 真正填上，兼容分层去重立刻生效；②在 crop 选择器（`orchestrator` novelty）加**状态新颖度**再排序：优先选与已存状态不同的 crop |
| 5 | 其他必要逻辑 | 见下 | 见下 |

### 其他必要逻辑（核心 3 的补强）

- **遮挡门控**：`occlusion` 已被分类但未参与门控 → heavy 遮挡按 `identity_visible=false` 处理（降级为非锚，不拒、保多样性）。随批量调用返回，0 额外调用。
- **描述升级**：现状描述只在 asset 无描述时写一次 → 允许"新状态/更清晰观测"升级 `asset.d`（确定性规则）。
- **入库端清晰度/尺寸复查**：感知层查过的清晰度/最小边，在记忆更新入库前做一次轻确定性复查，避免 bbox-only 兜底把弱锚写进库。
- **cohesion 门控②在 bench 生效**：需把 curator 的 embedder 从 `HashEmbedding` 换成 DINOv3（生产 `MemoryPolicy.production()` 已配 per-type floor）；离线默认仍自动空转。
- **写路径身份门控（VLM-first，已实现）**：`MemoryUpdater._reconcile_identity` 用编码器分 χ 做 shortlist 与高精度短路（χ≥β_τ+δ⁺→直接合并），
  清晰拒（χ≤β_τ−δ⁻→新建），只把 χ *灰区*交给 VLM 仲裁（`mllm/identity_judge.py`，温度 0、按模型标定 θ）；**强模糊 crop 一律 defer 为新的临时记录**（所有判定器在强模糊都会误合并）；
  判定器弃权/低置信则回落到确定性 χ≥β_τ。默认 `NullIdentityJudge` 弃权 → 决策与旧规则逐位一致；`policy.identity_vlm_enabled`（生产已开）+ 注入真 judge（`MEMSTRATA_IDENTITY_JUDGE=vlm`）才激活。
  依据见控制实验 `experiments/methods/MemStrata/20260725_vlm_vs_embedding_robustness`（通用编码器最弱、固定阈值脆、8B 强模糊 false-merge 炸→32B/θ=0.90 更稳），论文 §4.5 / Sec. exp-idgate。

### VLM 调用预算（读写不对称）

**读侧（每生成步，关乎响应速度）：**

| 读侧场景 | 慢思考调用 | 合计 |
|---|---|---|
| 全部名字命中（快思考） | 跳过 | **0** |
| 名字缺失/歧义，或需为新状态挑旧参考 | 1（意图解读慢路 / 参考挑选合并成一次） | **≤1** |

**写侧（每已实现 chunk，关乎记忆质量，可多但 lean）：**

| 写侧场景 | 调用 1（帧级发现+命名，R7b） | 调用 2（crop 级批量：属性+新实体验证+identity_visible） | 调用 3（身份仲裁，R9b，按需） | 合计 |
|---|---|---|---|---|
| 仅路径 A（老实体无新状态） | 跳过 | 跳过 | 跳过 | **0** |
| 有 D（自我发现），χ 均落短路/清晰拒/强模糊 | 1 | 1（批量所有候选 crop） | 0（编码器即决/ defer） | **2** |
| 有 D，且部分落 χ 灰区 | 1 | 1 | +k（k=灰区 discovered 观测数；每个一次调用，调用内带候选多张参考图） | **2+k** |
| 有 B/C（新状态/新实体） | 按需 | 1（同批量调用带目标 desc 做匹配） | 0（B/C 为点名，身份由命名锚定，不走重认） | **≤2** |

> 调用 1/2 与实体数量无关；调用 3 只对*自我发现 ∧ χ 灰区*触发，且编码器短路/清晰拒/强模糊 defer 已先滤掉绝大多数，
> 故 k 通常很小。确定性预筛把绝大多数无效 crop 挡在 VLM 之前；读侧优先 0/1 次，写侧 2 次/chunk 起、身份灰区按需 +k。

### 落地步骤（已批准，全部在 `methods/MemStrata` 内、不碰 bench 公平性）

1. **改名**：`skills/memory_curation` → `skills/memory_update`；类 `AssetCurator` → **`MemoryUpdater`**
   （保留 `AssetCurator = MemoryUpdater` 与 `memory_curation` shim 兼容旧 import）；`steps/curate.py` 转发不变；
   更新 `registry.toml`/README/纲领措辞（Stratified Update = 记忆更新）。
2. **记忆快照 exporter**：在 `memory_update` 下新增 `snapshot.py`，产出上文 `memstrata-memory-1.0` 的
   `memory.json` + 同根 `visual/<kind>s/<id>/states/<state>/*.png` + 每状态 `first_seen_sec`/`appearances`；每 chunk 原子写出。
3. **确定性门控补强**：`lib/crop_quality.py` 加过曝门；`crop_qa.audit_crop` 接入；GDINO 兜底护栏。
4. **一次批量 crop 调用**：新增批量结构化 MLLM 角色/客户端，一次返回
   {state, view, shot, lighting, occlusion, identity_visible, description, matches_target(desc)}；
   记忆更新与首见 seed 验证都消费它。
5. **状态新颖度**：`orchestrator` 选 crop 时加 state-aware 再排序；记忆更新确认 `compatible_stratum` 生效。
6. **意图解读挂"取证规划"（快慢结合）**：`intent_understanding` 为每个请求项标注 A/B/C/D 分支与快/慢档，
   驱动生成前的参考图选择与生成后的取证方式。
7. **校准 + 自检**：per-type 阈值按控制集重标；补方法侧 probe/smoke，跑 `pytest`。
8. **VLM-first 身份门控（已实现）**：`mllm/identity_judge.py`（Null/Heuristic/Vlm 三档，温度 0、json_schema）+
   `MemoryUpdater._reconcile_identity` 的 χ 分带（短路/清晰拒/灰区 VLM/强模糊 defer/弃权回落）+ `MemoryPolicy` 开关
   （`identity_vlm_enabled`/`identity_shortcircuit_margin`/`identity_gray_margin`/`identity_vlm_theta`/`identity_blur_*`，生产预设已开）；
   角色注册 R9b；单测 `tests/test_identity_gate.py`（8 例，含短路免调、灰区路由、θ 门、强模糊 defer）。默认离线逐位不变。

---

## docs 治理（已批准 / 本次一起执行）

`docs/` 目前仍是拆分前的混合态，含大量 **bench 专属**文档，迁往 `benchmarks/VMem-Bench/docs/`：

| 处置 | 目录/文件 | 说明 |
|---|---|---|
| **保留（方法侧）** | `docs/method/*`、`docs/overview/`、`docs/operations/`、`docs/glossary.md` | 更新路径引用（很多仍指向 `benchmarks/MemStrata/`） |
| **迁往 VMem-Bench** | `docs/benchmark/*`、`docs/baselines/*`、`docs/design/bench/*` | 这些是基准协议/baseline/公平性，属评测侧 |
| **分拣** | `docs/experiments/*` | 方法侧实验计划留下，bench 实验计划迁走 |
| **更新纲领路径** | `src/memstrata/docs/design_philosophy.md` | 顶部引用 `sut_design.md` / `benchmarks/MemStrata/docs/crop_principles.md` 已失效，改指 `docs/method/design.md` 与本包内 crop 契约 |

---

## 运行自检

```bash
python -m pytest -q
PYTHONPATH=src python3 -m memstrata.production.run --backend recording --decompose none --no-flux --no-autoserve --segments 2
PYTHONPATH=src python3 -m memstrata.production.run --backend oracle --decompose none --no-flux --no-autoserve --segments 2
```

真实 GPU 闭环（Wan / FLUX / Qwen）需要自备权重，见 [`docs/operations/models_and_environments.md`](docs/operations/models_and_environments.md)。论文表数字认 git 分支 `paper-reproduction`，不认随意的 `main`。仓库里部分历史文档仍写着旧 monorepo 路径（`methods/MemStrata`），在本仓里都指仓库根。
