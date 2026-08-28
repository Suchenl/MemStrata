# AGENTS.md · MemStrata

> 本文件是 MemStrata 方法仓的架构边界声明。触碰本仓代码前先读；它描述公开仓库的
> 自包含边界与和 VMem-Bench 的 JSON 交互方式。

## 规则 1：方法仓自包含

`src/memstrata/` 不得 import `vmem_bench`、`montage` 或任何本仓之外的项目代码。
第三方 Python 库和按路径加载的模型权重可以使用。运行产物可以写入用户指定的
外部磁盘，但配置、清单和入口必须由本仓代码提供。

## 规则 2：与 VMem-Bench 解耦

`memstrata` 与 `vmem_bench` 永远不能互相 import；两者只通过公开的 JSON 契约
（`PromptPacket`、`ObservationPacket`、`ComposedContextRecord`）交互，不共享进程内对象。

需要同时认识方法与基准的代码，只能放在 VMem-Bench 仓库的
`scripts/evaluate_baselines/` 下。这个目录包含 MemStrata adapter 与跨包集成测试；
不要把具体 benchmark 或 baseline import 塞进 `src/memstrata/`。

## 目录布局

```
.
├── src/memstrata/       # 唯一 Python 源码根
├── configs/             # 图像/视频后端 TOML
├── docs/                # 方法、运行和术语文档
├── production/          # 剧本与被忽略的运行产物
├── models/              # 本地权重路径（不提交权重）
└── templates/           # 可复制的配置/模板
```

## 运行自检

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest -q
bash scripts/memstrata/cpu_demo.sh
```

真实 GPU 运行需要用户自己的生成器、编码器、VLM 权重和 ffmpeg；配置方式见
`MODELS.md`。论文 Track A 的评测 adapter、gold 和 scorer 见公开的
[VMem-Bench](https://github.com/Suchenl/VMem-Bench)。

## 文档与兼容性

方法设计以 `docs/README.md`、`docs/method/` 和 `src/memstrata/docs/` 为准。
历史 checkout 的兼容路径只服务迁移，不是公开安装前提。变更用户流程时同步更新
`README.md` 与 `README.zh.md`；`paper-reproduction` 分支应保持论文协议冻结。
