# AGENTS.md · methods/MemStrata

> 本文件是 `methods/MemStrata/`（**MemStrata 方法包 `memstrata`**）的强制性架构边界声明，供任何要
> 触碰本目录代码的 agent（或人）在动手前先读。这是硬约束，不是建议；违反视为架构违规，而不是风格问题。
> 根目录 [`AGENTS.md`](../../AGENTS.md) 是整个仓库的入口，本文件是它在 `methods/MemStrata/` 这一层的
> 具体化；两者不冲突时，本子目录内的边界问题以本文件为准。
>
> **背景（2026-07 拆分）**：原 `benchmarks/MemStrata/` 已拆成两个各自独立、互不依赖的子项目：
> 方法包 `memstrata` 迁到本目录 `methods/MemStrata/`，配套评测基准迁到
> [`benchmarks/VMem-Bench/`](../../benchmarks/VMem-Bench/)（包名 `vmem_bench`）。下面的规则按这个新布局写。

## 规则 1：整体自包含——对 `methods/MemStrata/` 之外零依赖

`src/memstrata/` **不得 import 任何 `methods/MemStrata/` 之外的代码**，尤其不得 import
`src/montage/`（`montage.*`）和评测基准 `vmem_bench`（`vmem_bench.*` / 旧名 `memstrata_bench.*`）。
产出（生成视频、记忆库、run 输出、日志、校准文件等）也不得写到 `methods/MemStrata/` 之外的仓库路径。

- 第三方库（numpy / torch / transformers / PIL / …）与按路径加载的模型权重（统一走仓库
  `models/model_weights/`，见根目录 `AGENTS.md`）**不算耦合，允许使用**。
- 这条规则管的是**逻辑/代码引用边界**，不是**物理磁盘选择**：超大运行产物（视频、权重缓存）仍按
  `run-output-ledger` 技能走 `${ALLOWED_LOCAL_MEDIA_PATH:-.}` 系列磁盘的容量故障转移，跟本规则不冲突——前提是引用
  路径/清单文件本身仍然登记在 `methods/MemStrata/` 内部（不要把索引/清单也搬出去）。
- 为什么要这么严格：`memstrata` 要能作为独立方法子项目单独拆分发布，任何一处偷偷 import 了
  `src/montage/` 或 `vmem_bench`、或依赖仓库其它路径的产出，都会在拆分那一刻直接炸掉。

## 规则 2：不被外部引用，且永不 import 评测基准

`methods/MemStrata/` **不得被 `methods/MemStrata/` 之外的任何代码 import**（包括 `src/montage/`、
顶层 `experiments/`、`benchmarks/VMem-Bench/src/vmem_bench/`）。`memstrata` 与评测基准 `vmem_bench`
**永远不能互相 import**，无论哪个方向——两者只通过纯 JSON 契约
（`PromptPacket` / `ObservationPacket` / `ComposedContextRecord`）交互，不跨界共享 Python 代码、
不共用进程内对象。评测基准侧的对偶边界声明见
[`benchmarks/VMem-Bench/`](../../benchmarks/VMem-Bench/)。

## 规则 3：跨包代码只允许住在 VMem-Bench 的 `scripts/evaluate_baselines/`

任何需要**同时**认识方法与基准的代码——例如把 `memstrata` 当作 system-under-test / baseline 跑通
`vmem_bench` 的评分 harness、或把 MemStrata 和外部 baseline 一起对比——**只能**放在
[`benchmarks/VMem-Bench/scripts/evaluate_baselines/`](../../benchmarks/VMem-Bench/scripts/evaluate_baselines/)
（唯一被授权的跨包区），**永远不允许**出现在 `src/memstrata/` 或 `src/vmem_bench/` 的源码里。

- 跨包驱动脚本：`benchmarks/VMem-Bench/scripts/evaluate_baselines/trackA/memstrata/score_memstrata.py`
  （构造真实 memstrata SUT 并跑 vmem_bench 的评分 harness）。
- 跨包集成测试：`benchmarks/VMem-Bench/scripts/evaluate_baselines/tests/`（其 `conftest.py` 会把
  两个 `src/` 一并挂上 `sys.path`）。这些是唯一被允许同时 import `memstrata` 与 `vmem_bench` 的测试。
- 每新增一个要评测的外部 baseline，就在 `evaluate_baselines/` 下给它开独立目录/脚本，不要往
  `memstrata` 或 `vmem_bench` 的源码里塞任何具体 SUT 的 import 或 `--sut-variant` 分支。

## 目录布局（当前）

```
methods/MemStrata/
├── AGENTS.md                 # 本文件：架构边界声明
├── README.md                 # 方法总览（拆分后的措辞以本文件为准）
├── pytest.ini                # pythonpath=src, testpaths=src/memstrata
├── src/memstrata/            # ★ 方法包（唯一源码根）
│   ├── bank/                 #   记忆/资产库
│   ├── skills/               #   composition / decomposition / crop_acquisition / memory_* / optimization / …
│   ├── steps/                #   生产流水线步骤（intent / compose / curate / generate / keyframe / …）
│   ├── adapters/             #   bench.py = 对外 JSON 契约适配（不 import vmem_bench）
│   ├── encoders/  mllm/  lib/  extras/  cora_mllm/  production/
│   ├── docs/                 #   包内设计说明
│   └── tests/                # ★ 方法自检（+ skills/*/tests/），assert-based，禁 import vmem_bench
├── configs/                  # 图像/视频后端 TOML
├── docs/                     # 方法知识文档（先读 docs/README.md；docs/method/ 为权威设计）
├── experiments/              # 方法侧实验/probe（自包含）
├── production/               # 生产 run 产物 + 剧本（outputs/ 已 gitignore）
├── models/  outputs/         # 运行时权重/产物（gitignore）
└── _archive/                 # 历史原型（不参与运行；其中残留的旧 import 仅为归档，勿复活）
```

## 运行自检

```bash
cd methods/MemStrata
python -m pytest            # pythonpath=src, testpaths=src/memstrata
```

跨包（method×bench）集成测试不在此收集，见规则 3 的 `evaluate_baselines/tests/`。

## 约定

- 模型权重统一走仓库 `models/model_weights/`（见根目录 `AGENTS.md`）；从仓库拆分独立发布时再调整。
- 详细设计规格在 [`docs/method/philosophy.md`](docs/method/philosophy.md)（最高纲领）、
  [`docs/method/design.md`](docs/method/design.md)（含 planner fallback 必须保守、禁止返回全部资产等约束）；
  文档索引见 [`docs/README.md`](docs/README.md)。
- 上述文档如与本文件措辞冲突，以本文件为准；发现不一致请顺手同步修正，不要留两份互相矛盾的规则。
- 注：原 AGENTS 规则中的 `vlm_output` 续标/清场纪律属于**标注/基准**职责，已随基准迁至
  `benchmarks/VMem-Bench/`，不再由方法侧维护。
