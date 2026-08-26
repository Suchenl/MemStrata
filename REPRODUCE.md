# paper-reproduction：MemStrata 冻结核

本分支 **不是** 生产 `main`。它对应内部 git 快照

`VMem-Track-A-MemStrata` @ `51be2914`

也就是论文 **Track A Stage 1**（91 movies / MemStrata `__B16`）当时的方法代码树。
公开仓做了机械重命名与路径清洗，**没有改算法**。

| 分支 | 职责 |
|---|---|
| `main` | 生产：继续改方法，不保证对齐论文表 |
| `paper-reproduction`（本分支） | 冻住 Track A Stage 1 当时的 MemStrata 代码 |

论文数字只认本分支（建议再打 tag `paper-reproduction-v1`）。`pytest` **不能**复现论文表。

## 与 main 的区别

- 代码树来自 2026-07-28（北京时间）Track A Stage 1 快照，不是后来的生产提交。
- 默认剧本路径在树内：`production/screenplay/products/en/0000_detective_mystery.json`。
- 权重不进 git。未设置 `PUBLIC_MODELS_ROOT` 时，CPU `import` / `recording` 冒烟不得去碰真实权重。

## Track A vs Track B

- **Track A Stage 1（本仓职责）**：方法侧入口是 `python -m memstrata.production.run`；评测 adapter 在 **VMem-Bench** 仓的 `paper-reproduction`：
  `scripts/evaluate_baselines/trackA/baseline_adapters/causal/runner.py --adapter memstrata`
- **Track B 论文表（30 stories / system）**是后来在更新代码上跑的。本冻结核 **不声称**能对上 Track B 表。打分器在 VMem-Bench 仓，见那边的 `REPRODUCE.md`。

## 无 GPU 冒烟

```bash
export CUDA_VISIBLE_DEVICES=
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m memstrata.production.run --help
PYTHONPATH=src python -m memstrata.production.run --list-backends
PYTHONPATH=src python -m memstrata.production.run \
  --backend recording --decompose none --no-flux --no-autoserve --segments 2 \
  --outputs-root /tmp/opensource-paper-smoke/memstrata
```

`recording` / `oracle` 不跑真实生成器。不要省略 `--no-flux --no-autoserve --decompose none`，否则会去拉 FLUX / Qwen / crop server。

`oracle` 还需要配置里的源视频文件；缺视频时片段会被 SKIP，退出码仍可能是 0。无 GPU 冒烟请用 `recording`。

## 要对上 Track A 论文表，读者还缺什么

本仓只提供 **当时的方法代码**。还需要：

1. 同源的 **VMem-Bench `paper-reproduction`**（adapter + gold JSON + Stage 1 runner）。
2. **源视频**（Blender Open Movies + LSMDC）。本仓与 HF **都不发像素**。LSMDC 需自行申请。
3. 权重，经 `PUBLIC_MODELS_ROOT` 指向本地目录（SAM3 / DINOv3 / Qwen / 以及生产路径用的 Wan / FLUX——Stage 1 adapter 写路径主要是感知，不跑完整生成器）。
4. GPU。91 部电影的 Stage 1 不是 CPU 能做完的。
5. 环境变量：`PUBLIC_MODELS_ROOT`、`MEMSTRATA_SRC`（给 VMem-Bench adapter 指到本仓 `src/`）、`VMEM_DATASETS_ROOT`（视频根）。

没有把 91 部 run 产物或权重放进 git。在补齐上述外部依赖之前，**禁止声称已经对上论文数字**。
