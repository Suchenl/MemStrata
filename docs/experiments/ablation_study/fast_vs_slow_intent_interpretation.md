# Fast / Slow 意图解析消融

本文档登记一项计划中的 MemStrata 方法消融，不代表实验已经执行。只有可复现运行完成后，才能补充结果与结论。

## 研究问题

当用户意图表达不完整或存在歧义时，基于模型的慢速意图解析能否带来足够的资产选择收益，以抵消额外的延迟、模型调用与 token 成本？

## 对比变体

- `memstrata-fast`（论文默认）：使用确定性的 canonical name / alias 匹配，再通过字典直接解引用 memory bank；读路径模型调用数为 0。
- `memstrata-slow`：MLLM 读取下一 chunk 的 prompt，以及可用资产的名称、类型和描述，返回需要组合的 bank ID；未知 ID 会被过滤，空输出或调用失败时安全回退到 fast。

公开别名 `memstrata` 等价于 `memstrata-fast`。Slow 是显式消融变体，不能静默替换论文默认路径。

## Prompt 分桶

两种模式必须在完全相同的冻结 bank 状态和 prompt 顺序上比较：

1. 明确包含 canonical name；
2. 使用已登记 alias；
3. 使用代词或指代表达；
4. 省略名称，仅描述外观、状态或叙事意图；
5. 同类实体存在多个合理候选；
6. 明确包含否定、排除或 avoidance 意图。

第 1 桶验证 slow 不会破坏确定性契约；第 3–5 桶衡量 slow 的主要潜在收益。

## 指标

- selected-ID precision、recall 和 exact-set accuracy；
- MemStrata-Bench 的 Sufficiency、Parsimony、Fidelity、Avoidance 与总分；
- 单 prompt 延迟的 median / p95；
- 模型调用数、token 用量与估算成本；
- slow → fast fallback 比例及原因；
- 调用失败率与未知 ID 过滤数量。

## 控制变量与报告要求

- 使用相同的冻结 gold、memory bank 状态、prompt 顺序和 scoring 版本。
- 固定 planner 模型、服务端点、解码参数、超时与重试预算。
- canonical-name 桶和 ambiguous 桶必须分开报告，不能用总体均值掩盖 slow 的适用边界。
- 质量收益必须与延迟、模型调用和成本同时报告。
- 论文主结果默认使用 `memstrata-fast`；除非正式修改方法契约，`memstrata-slow` 只作为消融。

## 实现入口

- `src/memstrata/steps/intent.py`
- `src/memstrata/pipeline.py`
- `src/memstrata/adapters/bench.py`
- `scripts/memstrata/score_memstrata.py`

