# 运行交接：22 部 frozen BlenderOpenMovies × 全系统

> 给「跑实验的 agent」的自包含指令。**为什么这么跑**见
> [`baselines/fairness_decisions.md`](../../../../benchmarks/VMem-Bench/docs/baselines/fairness_decisions.md)；实验矩阵与 E 运行手册见
> [`fairness_experiment_plan.md`](../../../../benchmarks/VMem-Bench/docs/experiments/fairness_experiment_plan.md)。本文只讲**怎么跑 + 保留哪些产物**。

## 目标

在 **22 部已冻结的 BlenderOpenMovies**（有 `gold/chunk_index.json` 的；`sita_..._part1` 无 frozen
gold，不算）上，跑一遍**所有系统**（含我们的 MemStrata），把每个样本的结果**就地**写进该样本的
`benchmark_run/`，并完整保留指标计算产物与视觉检索结果，供论文放图与分析。

系统覆盖（主表）：
- MemStrata SUT：`memstrata-fast` + 4 个方法 ablation（`no_name_anchor / no_type_routing /
  no_dedup / no_avoidance`）。（`memstrata-slow` 需 MLLM planner，可选另跑。）
- 因果对手：`helios`、`longlive_rag`、`memflow`、`iamflow`（后三者各需每影片 GT trace）。
- 检索：`text_retrieval / frame_retrieval / textframe_fusion`，各报 `k ∈ {1,3,5,budget}`。
- 诊断：`full_history / recency_retrieval / sliding_window / selection_oracle`。
- **不跑** `decmem`（输入模态不匹配，见 D5）；scripted/agentic 系统不进定量主表。

## 环境（节点上）

```bash
cd .
export PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} PYTHONPATH=src
# 打分 embedder 的权重根：dinov3 走 PUBLIC_MODELS_ROOT；megaloc + siglip2 走仓库
# models/model_weights（子项目自带的 models/model_weights 目前为空）。不设这个，
# siglip2/megaloc 会静默降级、SigLIP2/Loc 列空着（正是下文验收门槛 2 要拦的坑）。
export MEMSTRATA_WEIGHTS_ROOT=${MEMSTRATA_WEIGHTS_ROOT}
export HF_HOME=${MEMSTRATA_WEIGHTS_ROOT}  # siglip2 缓存命中
PY=python3   # 节点无 repo .venv
```
`vmem_bench` import 时强制 `HF_HUB_OFFLINE=1`，所以 **siglip2-base-patch16-224 必须先在本地
缓存**（`$HF_HOME/hub/models--google--siglip2-base-patch16-224`）；缺就先一次性
`huggingface_hub.snapshot_download("google/siglip2-base-patch16-224")`（需临时联网），之后即可离线跑。
长任务在节点的 **tmux** 里跑（SSH 掉线不杀任务）；不要把前台挂在会随登录退出的 SSH 会话上。
> 节点 CPU 争抢（如他人的大批量 ffmpeg，`loadavg` 几十）会让打分（图像预处理 CPU-bound）慢几十倍；
> 铺开前先 `cat /proc/loadavg` 看一眼，必要时 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` 做好邻居。

## 第 1 步：生成因果 trace（写进 `<movie>/gold/`）

否则 `longlive_rag/memflow/iamflow` 行会诚实标 `skipped_no_trace`（**绝不伪造**）。当前只有 4 部齐全
（`big_buck_bunny / caminandes_2_gran_dillama / caminandes_3_llamigos / charge`），`cosmos_laundromat`
缺 iamflow，其余未开始。脚本逐 stage 幂等可续跑。

```bash
# (可选) 先从 vllm env 起 IAMFlow 的 LLM/VLM 服务器（DiT 仍留在 vace，忠实提速）
bash scripts/baselines/iamflow/servers/serve_iamflow_vllm.sh 0        # LLM :8100 + VLM :8101
curl -s http://127.0.0.1:8100/v1/models             # ready 后再跑

# 全影片、幂等；iamflow 走 vLLM + 分数缓存
CUDA_VISIBLE_DEVICES=0 $PY scripts/vmem_bench/compare/generate_causal_traces.py \
    --movies all --skip-complete \
    --iamflow-llm-endpoint http://127.0.0.1:8100/v1 \
    --iamflow-vlm-endpoint http://127.0.0.1:8101/v1 --iamflow-vlm-cache
```
多卡/多节点并行（墙钟≈最慢单片）：每个 GPU 单独起一个 `generate_causal_traces.py`
（或 `causal/runner.py`）进程，影片列表互斥。

## 第 2 步：跑全部系统 × 22 部影片

结果就地写进每个样本的 `<movie>/benchmark_run/`。

```bash
# name-anchored = 主设定（默认 --visual on / siglip2 / megaloc / 4 ablation / causal）
$PY scripts/vmem_bench/compare/run_movie_benchmark.py --all-blender
# description-only = 鲁棒性附录（name 从 prompt+observation 一并去除/改述，所有系统一视同仁）
# （CLI flag 仍是 --regime，A=name-anchored，B=description-only）
$PY scripts/vmem_bench/compare/run_movie_benchmark.py --all-blender --regime B
```
默认已开：`--visual`（headline = VisualFidelity）、`--extra-embedders siglip2`、
`--location-embedder megaloc`、`--include-causal`、`--ablations`。LSMDC 真人片再加
`--extra-embedders siglip2,arcface`。

## 必须保留的产物（供文章放图与分析）

全部在**每个样本的 `<movie>/benchmark_run/`** 下，请整目录保留，勿清理：

| 产物 | 路径 | 用途 |
|---|---|---|
| 排行榜（全行） | `benchmark_run/leaderboard.json` | 主表数据源 |
| 人读记分卡 | `benchmark_run/results.md` | 含 `VisFid / SigLIP2 / Loc` 列 |
| MemStrata 报告 | `benchmark_run/memstrata/<tag>_<emb>_report.json` | 6 指标 + `visual_fidelity`(+`_extra`/`location_fidelity`) + `id_diagnostics` + `efficiency` + `horizon_curve`/`memdist_curve` |
| **MemStrata 每-chunk 记录** | `benchmark_run/memstrata/<tag>_<emb>_records/` | **每 chunk 的选择逻辑(asset_id@rep)；定性图 + 指标复核必需** |
| baseline 报告 | `benchmark_run/baselines/<name>[_k{k}]/summary.json` + `score_report.json` | 逐 baseline / 逐 k 的指标 |
| **视觉检索结果** | `benchmark_run/visual_selections/<system>.json` + `by_chunk.json` | **每系统每 chunk 的选择 resolve 到具体 crop 路径 + prompt；跨系统并排视图 = 定性对比图的直接素材** |
| 运行日志 | `.../run.log`、`.../*.log` | 复现 / 排错 |
| 因果 trace + crops | `<movie>/gold/{gold_latents.pt, memflow_latents.pt, *_trace.json}` 及其指向的 crop 图 | trace 溯源 + 图里要显示的实际 crop |

- `visual_selections/`（由 `run_movie_benchmark` 自动调 `export_visual_selections.py` 生成）是
  **放图的核心**：已把选择 join 到 crop 路径（含相对/绝对），并给跨系统 `by_chunk.json` 并排视图。
- `*_records/` 是**指标可复核**的原始逐 chunk 决策。二者务必随每部影片一起留存。
- 跨影片汇总时从各 `leaderboard.json` 聚合，**不要改样本内文件**。

## ⚠️ 先验证再信任：第一次跑当作 smoke，不是终稿

**重要**：scoring embedder 缺 torch/GPU/权重或 API 不匹配时是**静默降级**（不报错）：
- DINOv3 headline 加载失败 → **悄悄退回 ID 组合 headline**，公平性修复失效；
- SigLIP2 / MegaLoc 失败 → 从 `extra` 丢掉 → `SigLIP2`/`Loc` **列空着**，无报错。

所以「run 成功」≠「数字是设计的那套」。**先只跑 1–2 部影片**（例如 `--movie-dir
data/BlenderOpenMovies/big_buck_bunny` 与一部 caminandes），逐项核对下面的**验收门槛**，全绿了再
`--all-blender` 铺开。信任任何数字前必须满足：

1. **headline 真的是 VisualFidelity**：每个 `*_report.json` / `summary.json` 里
   `headline_kind == "visual_composite"`，`versions.metric_version == "3.0.0-visual"`，
   `versions.scoring_embedder` 是 dinov3。不是就说明 DINOv3 没加载，**当前数字作废**。
2. **多 embedder 列非空**：`visual_fidelity_extra.siglip2` 有值、`location_fidelity` 有值；
   `results.md` 的 `SigLIP2`/`Loc` 列不是空白。真人片额外核 `arcface`。
3. **日志无静默降级**：`grep -ri "unavailable\|VisualFidelity disabled\|degrade" */run.log
   */*.log` 无命中。
4. **iamflow vLLM 忠实性抽查**：先对**同一部短片**分别用 HF（不带 endpoint）与 vLLM（带 endpoint）
   各跑一次 iamflow trace，核对 `iamflow_agent_trace.json` 的 `retrieved_source_latents` 基本一致
   （贪心解码应高度一致；若系统性偏差，先查 chat template / 图像编码，别铺开）。
5. **causal 覆盖 & 无伪造**：跑完后每部影片要么有真实 trace、要么 `skipped_no_trace`；
   `leaderboard.json` 里不得出现 `*_budget_proxy` 冒充方法行。
6. **无 OOM / 截断**：长片 RoPE 扩展与显存 OK（trace 日志无 OOM、无 `max-latents` 截断）。
7. **每系统 chunk 数一致**：同一影片下各系统 `num_chunks` 相同（同一份 frozen gold）。

任一不过 → 停下修，别把降级后的数字写进文章。

## 忠实性红线（不可违反）

- 缺 trace 的影片保持 `skipped_no_trace`，**绝不伪造行**。
- 不 truncate 历史、不换蒸馏/更小模型、不用 name/budget 代理当方法行。
- 因果对手的机制 forward 不能省（省了就是 `*_budget_proxy` 消融，非方法行）。
- `decmem` 不进主表（输入模态不匹配，H800 也解决不了）。
- IAMFlow 的 DiT/VAE 始终在本地 `vace` 进程跑（与已冻结 BBB trace 的 KV 数值一致）；vLLM 只承接
  LLM/VLM（同权重、贪心解码）。
