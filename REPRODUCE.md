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

## 安装检查（不跑论文、也不等于方法）

确认包能 import。对上 Track A 表仍然要 GPU + 权重，见下一节。

把本仓和 VMem-Bench 的 `paper-reproduction` **并排放**（`../VMem-Bench`）。然后：

```bash
python -m pip install -e ".[dev]"
python scripts/memstrata/doctor.py
bash scripts/memstrata/cpu_demo.sh
```

`cpu_demo.sh` 已带 `--backend recording --no-flux --no-autoserve --segments 2`。不要改回默认 Wan/FLUX，那会去拉权重。

## 要对上 Track A 论文表，读者还缺什么

本仓只提供 **当时的方法代码**。还需要：

1. 同源的 **VMem-Bench `paper-reproduction`**（adapter + gold JSON + Stage 1 runner）。
2. **源视频**：逐步获取说明见 VMem-Bench [`docs/DATA.md`](https://github.com/Suchenl/VMem-Bench/blob/paper-reproduction/docs/DATA.md)（并排放时是 `../VMem-Bench/docs/DATA.md`）。BBB：`bash scripts/prepare_blender.sh`。LSMDC：官方申请页 + 拼成 `LSMDC_Videos_Stitched/<movie_id>.mp4`。本仓与 HF **都不发像素**。
3. 权重：见 [`MODELS.md`](MODELS.md)。`python ../VMem-Bench/scripts/doctor.py` 会打印缺哪一项、以及对应的 `huggingface-cli` 命令。
4. GPU。91 部电影的 Stage 1 不是 CPU 能做完的。

Adapter 会在 `../MemStrata/src` 找到本包；只有目录名不是 `MemStrata` 时才需要 `MEMSTRATA_SRC`。

没有把 91 部 run 产物或权重放进 git。在补齐上述外部依赖之前，**禁止声称已经对上论文数字**。

## Citation

```bibtex
@article{chen2026memstrata,
  title={Stratifying and Benchmarking Long-Range Memory for Causal Long Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua},
  year={2026}
}
```

See [`CITATION.cff`](CITATION.cff).

