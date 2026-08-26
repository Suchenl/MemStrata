# 《灯塔守望录》(`0001_lighthouse_keeper`) — 长程记忆 Hard-Case 刻意设计

本剧本不是为了"好看的故事"而写，而是为了**在真实生成循环里逼出 MemStrata 外部记忆机制的失败模式**。它跨越数十年、10 个场景 / 6 个地点 / 16 个 shot、11 个实体，并在时间轴上**故意**埋入五类硬样本。下表把每个硬样本对应到具体 shot / chunk / 实体，方便复现和读表。

> chunk 索引 = shot 顺序号 - 1（`iter_shots` 从 0 计）。shot_0001→chunk 0，…，shot_0016→chunk 15。

## 实体清单

| id | 名称 | 类型 | 记忆角色 |
|---|---|---|---|
| E1 | 伊莱亚斯 | character | 主线角色，跨年龄状态变化 + 结尾废弃 |
| E2 | 玛拉 | character | 长时缺席后回归 + 成长状态变化；与 E6 构成错认对 |
| E3 | 航海日志 | object | 长时缺席后回归 + 完好→风化状态变化 |
| E4 | 菲涅尔透镜 | object | 完好→开裂状态变化 + 废弃旧态 |
| E5 | 海燕号 | object | 沉没后废弃（avoid） |
| E6 | 莉娜（双胞胎） | character | 错认干扰项（wrong-instance） |
| E7 | 灯塔 | location | 场景回访锚点 |
| E8 | 灯室 | location | 场景回访锚点（S1/S4/S7/S9） |
| E9 | 礁石海岸 | location | 场景 |
| E10 | 渔村集市 | location | 错认场景 |
| E11 | 纪念铜牌 | object | 结尾，指代已故 E1（间接引用） |

---

## 硬样本 1 — 长时缺席后回归（Long-gap Re-appearance / Long-gap Recall）

**测什么**：一个实体在被点名后长时间不出现，很久之后重新点名，记忆能否正确把它调回来（而不是丢失、或退化成"最近活跃"的近因偏置）。

| 实体 | 出现 chunk | 缺席跨度 | 回归 chunk |
|---|---|---|---|
| **E2 玛拉** | 2,3,4,6 | **chunk 7–10 连续缺席（S5 独航 / S6 风暴 / S7 裂痕）** | 11（S8 归来） |
| **E3 航海日志** | 1,6 | chunk 7–9、11–12 缺席 | 13（S9，且以**风化态**回归） |

- 玛拉的缺席横跨 3 个场景、4 个 chunk，回归时还叠加了"成长"状态变化——**同时考验召回 + 状态更新**。
- 这是 read-path 的核心压力点：MemStrata 默认按 name/alias 解引用，长缺席不应导致召回失败或读延迟随距离上升（对应指标 Return Success / Long-gap Recall / Latency Slope）。

## 硬样本 2 — 错认（Wrong-instance / 相似实例辨识）

**测什么**：两个视觉高度相似的实体，记忆是否会拿错人。

- **E2 玛拉** 与其**双胞胎姐姐 E6 莉娜**面容极似；靠可辨特征区分：**玛拉系深蓝头巾，莉娜系赭红披肩 + 贝壳发夹**。
- 莉娜**只在 chunk 5（S3 渔村集市）出现一次**，此后不再露面。玛拉在其前后大量出现。
- 陷阱：decompose 取 crop 时若只按"最像库里那个人"选，极易把莉娜误并入玛拉的身份，或反之。**身份门（DINOv3 exemplar）应确认"是不是玛拉本人"，而非"最相似"**（对应指标 Wrong-instance Rate）。

## 硬样本 3 — 状态变化（State-change / State Correctness）

**测什么**：同一实体的外观随剧情改变后，后续是否使用**当前状态**而非旧态。

| 实体 | 旧态 | 新态 | 变化点 |
|---|---|---|---|
| **E1 伊莱亚斯** | 中年灰白胡须 | 年迈(S7)→更苍老佝偻(S8) | 渐变 |
| **E3 航海日志** | 整洁黄铜包角 | 盐渍风化、封皮卷曲 | 暴风雨后（S9，chunk 13） |
| **E4 菲涅尔透镜** | 完好旋转 | 开裂、光束带缺口 | **S6 chunk 9（`operation=transform`）** |

- 透镜在 shot_0010（chunk 9）显式标 `operation=transform`，是状态迁移的明确信号；此后 S7/S9 的 continuity_requirements 要求**必须带裂纹**。
- 考验 curate 的状态事件写入 + compose 的"取当前态而非 deprecated 态"（对应指标 State Correctness）。

## 硬样本 4 — 废弃证据规避（Deprecated-evidence Avoidance）

**测什么**：某实体/某旧态被剧情废弃后，后续是否**不再复用**它。

- **E5 海燕号**：S6（chunk 9）沉没；此后 **chunk 10（S7）、chunk 11（S8）显式标 `operation=avoid`** → `iter_shots` 归入 `forbidden_ids`。之后不得再出现完好海燕号。
- **E1 伊莱亚斯**：S10（chunk 15）已故，**标 `operation=avoid`**，仅由 **E11 纪念铜牌**间接指代（见硬样本 5）。
- 考验 compose 是否把 deprecated rep 放入 `exclusions` 并从条件里剔除（对应指标 Avoidance Violation）。

## 硬样本 5 — 间接指代 / 缺席在场（Indirect Reference）

**测什么**：一个实体本体已不在场，但通过另一个物件被"提及"，记忆与生成应体现"缺席"而非把本体重新画出来。

- 结尾 S10（chunk 15）：伊莱亚斯（E1）已故、标 `avoid`，画面只出现**刻着他名字的纪念铜牌 E11**。E1 被引用但不应出现在画面里。这是"deprecated + 间接指代"的复合样本。

---

## 场景回访（Scene-return，贯穿全片）

| 地点 | 出现 chunk |
|---|---|
| 灯室 E8 | 0,1（S1）· 6（S4）· 10（S7）· 13,14（S9） |
| 灯塔 E7 | 8（S6）· 11,12（S8）· 15（S10） |
| 礁石海岸 E9 | 2,3（S2） |
| 渔村集市 E10 | 4,5（S3） |

- 10 个 scene-start（`is_scene_start=True`）会触发生成路由偏向 `recompose_keyframe`（跨场景不能 continue_ar）；同一地点多次回访考验场景锚点（location asset）的一致性复用。

## 时间跳跃（Time-skip）

S7→S8 之间是数年时间跳跃。它把"长时缺席"和"状态变化"叠加放大：玛拉长大、伊莱亚斯更老、日志风化、透镜仍裂——**一次跳跃同时激活硬样本 1 + 3**，是最接近真实长片的极端点。

---

## 与指标的对应（Track-B 一致性/速度指标）

| Hard case | 主要指标 |
|---|---|
| 1 长时缺席回归 | Return Success · Long-gap Recall（按缺席长度分桶）· Read Latency / Latency Slope |
| 2 错认 | Wrong-instance Rate |
| 3 状态变化 | State Correctness |
| 4 废弃规避 | Avoidance Violation |
| 5 间接指代 | Avoidance Violation（本体不复现）+ 引用解析正确性 |
| 场景回访 | 场景锚点一致性 · Read Latency 不随历史增长 |

## 复现

```bash
# 校验 shots / 硬样本接线（不生成）
PYTHONPATH=src python -c "
from memstrata.adapters.screenplay import load_screenplay, iter_shots
sp=load_screenplay('data/Screenplay/products/cn/0001_lighthouse_keeper.json')
shots=iter_shots(sp)
apps=lambda e:[s.chunk_id for s in shots if e in s.referenced_entities]
print('E2', apps('E2')); print('E3', apps('E3')); print('E6', apps('E6'))
print('avoid', [(s.chunk_id,s.forbidden_ids) for s in shots if s.forbidden_ids])
"

# 正片生产（默认 lightx2v + FLUX 关键帧）
bash scripts/memstrata/run_production.sh \
  data/Screenplay/products/cn/0001_lighthouse_keeper.json
```
