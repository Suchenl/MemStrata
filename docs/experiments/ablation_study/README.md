# MemStrata 消融实验候选

本目录统一记录所有“可能会做”的 MemStrata 消融实验，作用是保存研究问题、对比变体、控制变量、指标和执行边界，避免想法散落在聊天或临时日志中。

约定：

1. 每个消融实验单独一个 Markdown 文件，文件名直接描述实验主题。
2. 这里记录的是实验计划，不是实验结果；未实际运行时禁止填写虚构数字或结论。
3. 真正执行的实验必须遵守仓库实验规范，代码与配置进入 `experiments/`，结果进入对应 results 目录，并在 `experiments/REGISTRY.md` 登记。
4. 论文只能引用完成可复现运行且通过结果审计的消融结果。
5. 实验计划废弃时，应在原文件中写明原因，不要静默删除失败方向。

当前候选：

- [`fast_vs_slow_intent_interpretation.md`](fast_vs_slow_intent_interpretation.md)：比较默认确定性 fast 与显式 MLLM slow 意图解析的质量、适用边界和成本。
