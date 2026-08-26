# Terminology（统一术语，向 abstract/intro 对齐）

原则：**没有创新点的概念一律用通用术语；只有真正的 contribution 才用专名。** 全文（method/
related/experiments/conclusion/appendix/图表/notation）必须与 abstract、introduction 一致。

| 层级 | 规范术语 | 说明 / 反例 |
| --- | --- | --- |
| 方法名 | **MemStrata**（宏 `\method`，= **Mem**ory **Strata**） | 不要再用 "StrataMem" / "AutoFilm" / "Montage"（Montage 是项目/仓库名，不是方法名）|
| 核心机制 | **active composition over role-aware assets** | 把被动检索变成主动组合；不要叫 "memory retrieval" |
| 核心对象 | **role-aware asset**（role = 条件功能：identity anchor / scene / style / motion / negative …）| 这是 contribution，保留 |
| 持久状态 | **asset bank** $\mathcal{A}_n$ | 通用词；**不要**用 "production memory" / "production asset space" |
| 单个资产 | **asset** $a_j$，含多个 **purpose-specific representations** $\mathcal{R}_j$ | — |
| 资产关系 | **typed relations**（折叠进资产的 $\rho_j$）| **不要**单列 "asset graph $\mathcal{G}_n$" 作为独立组件 |
| 全局约束 | **global constraints / text assets** | **不要**用 "production bible $\mathcal{B}$"；它就是若干文本/规格资产 |
| 原始留存 | **archive**（整片/原始输出/日志/切片）| 通用词；不要叫 "production archive" |
| 每步生成条件 | **Composed Context** $\mathcal{C}_n$ | contribution；**不要**用 "asset package" / "composition package" |
| 组合方式 | **model-free composition**（确定性解引用，不调模型）| — |
| 生命周期 | **lifecycle status** $\psi$：candidate/reusable/used/rejected/deprecated/failed | — |
| 更新机制 | **condition-aware curation** | 通用描述即可，不必专名 |
| 生成器 | **reference-conditioned continuation generator**（黑盒、生成器无关）| — |
| 评测 | **MemStrata-Bench**，四轴：asset selection / functional-role assignment / constraint satisfaction / negative-asset avoidance | 不要叫 "asset composition evaluation" 作专名 |

被废弃的旧术语（全文不得再出现）：StrataMem、production memory、production bible、asset relation graph
$\mathcal{G}_n$、asset package / composition package、production archive、AutoFilm、LVG-Agent、
以及 LVG-Agent 时期的 latent/memory-record 记号（$m_n=(k_n,v_n,u_n,a_n)$、$\hat{\mathcal{R}}_n$ 等）。
