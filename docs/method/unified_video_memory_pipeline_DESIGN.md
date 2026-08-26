# 统一视频记忆管线 · Instance Cache 设计（DRAFT，待审）

> 状态：**提案 / 跑后迁移项目（未实现）**。⚠️ **禁止在今晚 Track A+B run 之前实现本设计**：
> 两轮设计评审 + 一次代码级评审确认，落地会动到 `IntentInterpreter` 返回类型、`AssetBank` schema、
> 读/写时序、抠图后端与 Track A 冻结契约（`contract.py`），爆炸半径过大。**今晚用现状代码按原样跑**，
> 本设计作为 run 之后的迁移工程（带兼容 shim + 针对性测试）分步落地。
>
> 本文件定义 MemStrata 的**统一记忆管线**：
> 以"视频为真相、按需物化"为哲学，以**一份统一 JSON plan** 为读/写两侧的共同契约。
> 关联：`philosophy.md`（库质量最高纲领，本文件不与其冲突）、`design.md`（包结构/接线）、
> `hybrid_retrieval_selfaudit_DESIGN.md`（读侧语义召回 + 帧级自审，本文件的检索/自审细节承接它）、
> `skills/intent_understanding/`（读侧 fast/slow）、`skills/memory_update/curator.py`（写侧物化）、
> `skills/crop_acquisition/`（WeDetect-Ref grounding）、`skills/memory_update/snapshot.py`（导出）。
> 与本纲领冲突时以 `philosophy.md` 六公理为准，本文件负责"管线与契约"，不改库质量目标函数。

---

## 0. 一句话

把长视频记忆管理**重述为一个 Instance Cache**：`long_video.mp4` 是**真相后端（backing store）**，
结构化记忆（JSON + crop + embedding）是**按需物化的缓存**；每一步生成意图先被解析成**一份统一 JSON plan**，
再由确定性执行器把 plan 变成"取哪条实例证据 / 落哪条新证据"。

方法名仍是 **MemStrata**；**Instance Cache 是它实现的抽象框架**（论文里加一层 framing，不改方法名、不改标题）。

---

## 1. 核心命题：Video-as-backing-store + 按需物化

记忆库不是"帧的仓库"，也不是"一次性全量抽好的实体表"，而是**建立在视频后端之上、按需填充的缓存**。
这直接给了我们一套 cache 词汇（论文叙事 + 工程语义双赢）：

| Cache 概念 | Instance Cache 对应 |
|---|---|
| Backing store（真相） | `long_video.mp4` + **历次意图 JSON plan（图，§7）** |
| Cache line | 一条已物化的**实例-状态**：`crop(s) + 描述 + embedding + provenance` |
| Cache hit | `name`/`alias`/**已确认的** semantic 命中已物化状态 → 直接解引用 |
| Cache miss | 历史见过但未入库 → **懒读现场物化**（`op:read`+WeDetect-Ref，§4.4/§5）；全新实体走生成后 write-through |
| Write-through | 生成出新状态后**立即物化入库**（读到才补的走 lazy） |
| Invalidation | `deprecate`（状态被取代/失效；**只走控制通道，模型不产出**，§2.5） |

**懒物化（acquire-on-reference）**：只有**被点名/被意图引用**的实体才实例化并入库；
自发现的边角（背景、字幕、天空…）默认不落库，避免碎片与噪声（沿用 `philosophy.md` 公理 2/4）。

---

## 2. 统一 JSON plan（本设计的核心）

> 定好这份 JSON 是重中之重：它是读/写两侧的**唯一契约**，也是 slow-path 语言模型输出的**受约束目标**。

### 2.1 为什么是一份 JSON（两个形态）

一份 plan 走两个形态：**IntentPlanV1**（意图层，§2.2，唯一需要生成/手写的部分）→ 执行器补成
**ResolvedPlanV1**（+解析/执行字段，§2.3）。**只有 IntentPlanV1** 面向"谁来填"，能被四种来源产出，
把"意图"从"生产意图"升维成"**以记忆为核心的实例管理**"：

1. **人填**——直接手写 IntentPlan，拿到最大控制权。
2. **规则填**——`instances × prompt` 名字匹配，填出稀疏的 fast IntentPlan（§3.1）。
3. **agent/MLLM 填**——slow 意图理解，填出丰满 IntentPlan（§3.2）。
4. **小模型 / guided decoding 规整**——用受约束解码（`response_format: json_schema, strict`）
   或一个小参数量工具模型兜底格式，避免人/MLLM 填出格式错误。

> **fast 与 slow 产出的是同一份 IntentPlan**：fast 更稀疏，slow 更丰满；执行器统一补全解析层，只写一次。

### 2.2 IntentPlanV1（v1.0，**唯一需要 MLLM 生成**的形态；guided-decoding 严格目标）

> 只含"意图层"——**没有** `entity_id`/`confidence`/`source`/`step_id`/`control`（那些是执行器/orchestrator 补的）。
> 场景 `raw_intent = 淋湿的红裙女孩把苹果递给小明`：①红裙女孩(老实体换湿状态 `update`)、
> ②小明(老实体只取 `read`)、③苹果(历史里出现过、尚未入库 → **懒物化** `read`)。

```jsonc
{
  "schema_version": "1.0",
  "raw_intent": "淋湿的红裙女孩把苹果递给小明",
  "mode": "slow",                               // "fast" | "slow"
  "targets": [
    {
      "key": "t1",                               // 本 plan 内本地标识；relations 用它引用（避免 ref 歧义）
      "ref": "红裙女孩",                          // 指代：名字 或 描述
      "kind": "character",                        // character | prop | location
      "op": "update",                             // read | create | update
      "state": {"select": "model_decided"},       // 判别式对象：{select:latest}|{select:by_state,state_type}|{select:model_decided}
      "write": {                                  // 落新态；op:read 时整块 null
        "state_type": "changed",                  // 闭集 default|changed|damaged|unknown（unknown 仅不确定时）
        "state_description": "淋雨后湿透、头发贴额头",// 该状态的瞬时描述（喂 state embedding）
        "identity_description": null,             // 仅 op:create 填：稳定身份文本（喂 coref，永不被 update 覆盖）
        "expected_angles": ["front"]              // 期望落哪些角度 crop；可空数组=不限
      },
      "materialize": {"policy": "if_missing", "grounding_query": null}, // 意图层只给策略，不给 need/source
      "output_budget": null                       // 可选覆盖；null=compose 默认（单位=rep 条数）
    },
    {
      "key": "t2", "ref": "小明", "kind": "character", "op": "read",
      "state": {"select": "latest"}, "write": null,
      "materialize": {"policy": "if_missing", "grounding_query": null}, "output_budget": null
    },
    {
      "key": "t3", "ref": "苹果", "kind": "prop", "op": "read",
      "state": {"select": "latest"}, "write": null,
      "materialize": {"policy": "if_missing", "grounding_query": "红色的苹果"}, "output_budget": null
    }
  ],
  "relations": [                                  // 按 target.key 引用（无歧义）
    {"subj": "t1", "verb": "hand_to", "obj": "t2", "instrument": "t3"}
  ]
}
```

`state` 是**判别式联合**（每种都是对象，无裸字符串）：`{"select":"latest"}` |
`{"select":"by_state","state_type":"changed"}` | `{"select":"model_decided"}` | `null`（仅 `op:create`）。

> `op:read` → `write=null`；`op:update` → `state`+`write` 都非空；`op:create` → `state=null`、
> `write.identity_description` 必填、`materialize.policy` 通常 `never`（全新实体只能生成后 write-through，§4.4）。

### 2.3 ResolvedPlanV1（执行器/orchestrator 补全，= IntentPlan + 系统字段）

同一份对象被**逐步补全**，MLLM 从不产出下列字段：

```jsonc
{
  // …IntentPlanV1 全部字段原样保留…
  "step_id": "step_017",                  // 〈orchestrator〉§7 图节点 id
  "parent_step_id": "step_016",           // 〈orchestrator〉时序边
  "produced_segment_id": "seg_017",       // 〈executor〉本步产出段 artifact
  "targets": [
    {
      "key": "t1", /* …意图层字段原样… */
      "resolve": {                         // 〈executor〉指代解析（此例：semantic 已过确认门）
        "match_kind": "semantic",          // name | alias | semantic | none
        "candidates": [                    // semantic 先给候选
          {"entity_id": "girl_01", "score": 0.82},
          {"entity_id": "girl_07", "score": 0.79}
        ],
        "needs_confirmation": false,       // 确认门已通过（未过时保持 true）
        "entity_id": "girl_01"             // ★ 仅 name/alias/确认通过后才 non-null（§2.4 红线）
      },
      "state": {"select": "model_decided", "state_id": "girl_01/changed#3"}, // 〈executor 二次调用〉回填
      "materialize": {"policy": "if_missing", "grounding_query": null,
                      "need": false, "source": null},   // 〈executor〉命中/未命中 + 源
      "status": "ok",                      // ok | materialize_failed | unresolved
      "failure_reason": null               // status≠ok 时填
    }
    // t2 / t3 同构
    // 未过确认门时：entity_id=null、state_id=null、status="unresolved"（不解析/不提交）
  ],
  "control": {"deprecate": []}             // 〈人/规则〉不在 IntentPlan 内
}
```

### 2.4 字段语义要点

- **★ `semantic` 不做身份分配（`philosophy.md` §3 红线）**：向量召回只产出 `resolve.candidates`
  （+ `needs_confirmation`），**绝不**直接写 `entity_id`。最终身份只由 `name`/显式 `alias`，
  或一道**确认门**（§4.1：新开上下文的 MLLM 复核 / 人工）敲定；不过则当**新实体**（转 `op:create`）。
- **state = 闭集 `state_type∈{default,changed,damaged,unknown}`（对齐 `design.md`）+ 自由文本 `state_description`**；
  `state.select` 是判别式对象（`latest`/`{by_state,state_type}`/`model_decided`），不拼字符串。
  （旧的 `state_label="wet"` 这种开放标签废弃：语义收进 `state_description`，桶键仍是闭集。）
- **`identity_description`（稳定，喂 coref embedding，仅 `create` 建立）vs `state_description`（每态、瞬时）分离**；
  `op:update` **只写 `state_description`，绝不覆盖 `identity_description`**——否则共指会随换装/换态漂移。
- **`materialize`：意图层只给 `policy`（`if_missing`/`never`/`force`）+ `grounding_query`**；
  `need`/`source` 由执行器按 cache 命中与 §7 图查询回填（命不命中 MLLM 无从知道）。
- **`state.select=model_decided`（§4.3）**：执行器给出该实例**已存 `state_id` 枚举**，一次
  `temperature=0`、**留痕、可复现**的受约束调用从中**选一个**回填；异常回退 `latest`。
- **C｜`kind`**：`character/prop` 走 WeDetect-Ref 框裁；`location` 物化用**整帧**而非框裁。
- **B｜图边**：`step_id`/`parent_step_id`（orchestrator）+ `produced_segment_id`（executor）
  让每份 plan 成为 §7"时间×实体图"的**一等节点**，零额外成本支撑长视频定位。
- **E｜`relations`**：按 `target.key` 引用（无歧义，含重复 ref/道具）；`verb` 为**可选自由文本**，
  compose 只需"同组共现"（`philosophy.md` 的"关系扩展的连续性证据"，预算紧张时最先砍）。*需 compose 侧读取才生效*。
- **F｜`resolve.candidates`**：语义匹配的 top-k 候选+分数，供审计与"top1/top2 差 < ε → 升级确认"的钩子。
- **`output_budget`**：可选逐 target 覆盖，单位 = **rep（crop）条数**；默认走 compose 的 `context_rep_budget`。
  方法侧**无**全局人为预算（`total_budget`/`as_of` 是评测护栏，不进本 schema）。
- **`status`**：per-target 执行结果 `ok|materialize_failed|unresolved`；**成功的 target 照常提交，
  单个失败不阻塞全步**，失败项带 `failure_reason`。
- **`key`**：plan 内唯一（建议 `^t\d+$`），仅本地寻址（relations 引用），不跨 plan。
- **`materialize.grounding_query` 为 `null`** 时执行器默认用 `ref` 文本作为 grounding 查询串。

### 2.5 deprecate 单独走"控制通道"（安全红线）

`deprecate` **不在 IntentPlanV1 内**（MLLM 严格 schema 根本没有这个位），只出现在 ResolvedPlan 的
`control.deprecate`，且**只允许人/规则写**。item 形如
`{"scope":"entity|state|rep","target_id":…,"reason":…,"replaced_by":…,"actor":…}`；
执行永远"标废弃、留痕、可回退"，绝不静默物理删除（沿用 `philosophy.md` §4）。

---

## 3. 读侧：fast / slow 两种可切换模式（都产出同一 plan）

> 两模式产出的都是 **IntentPlanV1**（§2.2）；执行器再统一补成 ResolvedPlan（§2.3）。

### 3.1 fast（确定性、model-free）
- 输入：用户写清的名字。
- 过程：规则填 IntentPlan（`op:read`、`state.select:latest`、`ref=名字`）；执行器 `name_match` 回填 `resolve`。
- 产出：稀疏 IntentPlan；这是"cache hit 快路径"，无模型调用。

### 3.2 slow（纯文本意图理解 → LLM 出 IntentPlan）
- 输入：**（库内实例清单 + 新意图）纯文本**，不喂像素。
- 过程：LLM 在 guided decoding 下产出丰满 IntentPlan；执行器做 **JSON↔JSON 双向匹配**（§4.1）补 `resolve`。
- 能力：可解描述指代（走 `semantic` 候选+确认）、可判 `create/update`、可要求 `model_decided` 选参考态。

> 两模式**互斥可切换**，不强制级联；但 slow 内部若名字够得着，仍优先用确定性 `name/alias`，
> 只有落空才动用 `semantic`（省 embedding 调用）。

---

## 4. 执行器：把 plan 变成动作（确定性优先）

### 4.1 双向匹配 + 确认门 = JSON↔JSON（守住红线）
把库内实例也表示为 JSON，`target` 与库内实例**两向互证**（新→库、库→新）：
`name/alias`（确定性，直接定 `entity_id`）→ `semantic`（向量召回，**只填 `candidates`+`needs_confirmation`**）→ `none`。
**semantic 命中不等于身份确定**：需过一道**确认门**——`needs_confirmation` 为真时，用**新开上下文**的
MLLM 复核（或人工）在候选间敲定；确认通过才写 `entity_id`，否则转 `op:create` 当新实体。
**身份分配始终由命名/显式别名/确认门锚定，embedding 只做候选召回**（`philosophy.md` §3，不越界成"重新决定这是谁"）。

### 4.2 `op:read`
解引用取该实例 `state.select` 指定状态的 crop（`output_budget` 为空时走 compose 默认上限）。model-free。
若该状态未物化（cache miss）且 `materialize.policy≠never`，先走 §5 从历史现场物化再返回（**懒读**）。

### 4.3 `op:update`（选参考态 → 生成 → write-through 落新态）
1. **选参考态（读 `state`）**：`state.select=model_decided` 时，执行器把该实例**已存 `state_id` 枚举**
   喂给模型做一次 `temperature=0`、**留痕可复现**的受约束选择，回填 `state_id`；异常回退 `latest`。
2. 下游用该参考态生成新外观。
3. **write-through（落 `write`）**：按 `write.{state_type, state_description, expected_angles}`
   把生成结果**立即物化为新状态**入库（§5）；**`identity_description` 不被覆盖**，并记录时间轴锚点。

### 4.4 `op:create` vs 懒物化（两种"库里还没有"，别混）
- **懒物化（`op:read`）**：实体在**历史 `long_video` 里出现过**、只是没入库 → 用
  `materialize.{policy=if_missing, grounding_query}` 从历史帧现场裁并落库（§5）。这是"已见但未缓存"。
- **`op:create`（全新实体）**：实体**历史里从没有** → **不能从历史裁**（`materialize.policy=never`），
  只能等本步**生成产出 `produced_segment_id` 后 write-through** 落库，`write.identity_description` 建立稳定身份。

### 4.5 `relations` → compose 共现上下文
`relations` 非空时，compose 把同一交互内的实体（按 `subj/obj/instrument` 的 `key` 解引用）**成组**给下游，
作为"关系扩展的连续性证据"（预算紧张时最先被砍，`philosophy.md` 公理 6）。

### 4.6 部分失败语义
per-target `status`：`ok` / `materialize_failed` / `unresolved`。**成功的 target 照常提交下游**，
单个失败**不阻塞全步**；失败项留 `failure_reason` 供审计与后续补跑。

---

## 5. 写侧 / 物化：从 backing store 到 cache line

- **crop 获取改用 WeDetect-Ref（referring grounding），替代 SAM3 概念分割**：
  直接用**实体描述串**在帧上定位并**框裁**（不做分割，省时且更稳；见 owner 决定）。
  常驻服务形态，避免每步重载模型。
- **质量门沿用 `philosophy.md` 的 WHO-before-WHERE 三闸 + 库级 cohesion 自审**：
  暗/低信息确定性门禁、identity_visible、embedding 内聚度；不误伤"高多样"。
- **一条 cache line 落**：`crop(s)` + **`identity_description`（实例级、稳定）** +
  **`state_description`（状态级、瞬时）** + **`identity_description` 的文本 emb**（喂 coref）
  + **视觉 emb**（§6）+ `provenance`（源帧、时间戳、frame_idx、生成/物化来源）。
  身份文本在 `create` 时建立、**`update` 永不覆盖**；状态文本每个新态各存一份。
- 源帧与 embedding **旁挂持久化**（`embeddings.npz` / `frames/`），**不塞进人读的 `memory.json`**
  （承接 `hybrid_retrieval_selfaudit_DESIGN.md` §3-C/§4）。

---

## 6. Embedding 栈与三类检索（输入对不同，别混）

预存策略：**物化时就抽好 embedding 写进 cache line**（旁挂 npz），检索 = query emb vs 库内 emb 的 cosine top-k。

| 用途 | 触发时机 | 输入A（query） | 输入B（库内） | 模型 |
|---|---|---|---|---|
| **共指候选召回** | slow 意图步（只有文本） | 新实体**描述文本** emb | 每实例**`identity_description`** emb（稳定，非状态文本） | `Qwen/Qwen3-Embedding-4B` |
| **语义读取（描述找对象）** | 名字/别名落空时 | 描述文本 emb | 库内实例 `identity_description` emb（text↔text） | `Qwen/Qwen3-Embedding-4B` |
| **视觉去重（同类不同个体）** | 物化/自审时（有像素） | 新 crop**视觉** emb | 库内 crop**视觉** emb（实例级判别） | `facebook/dinov3-vitb16` |

> **决策（owner 已定）**：内部生成场景下每个实例都会存描述，"描述找对象"用 **text↔text 就够**，
> **不上**跨模态 `Qwen3-VL-Embedding`，省一个模型。故 embedding 栈只有两个：
> `Qwen3-Embedding-4B`（文本共指/语义读取）+ `DINOv3`（视觉去重）。
> 二者始终是**门控/召回**，不做身份分配。

---

## 7. long_video 定位：意图 JSON 图（内部生成专用）

因为**每段视频都是我们按意图 JSON 自己生成的**，历次 plan 全部保留，天然构成一张
**时间 × 实体的图**：某实例在何时出现、以哪份意图生成，**查图即得**，无需复杂时序 grounding、无需扫像素。

- plan 保留 → 图的边（实例↔段↔时间）自动成形。
- 定位"t=x 的某实例最新状态" = 图查询 + cache 解引用。
- **外部导入视频不在本设计范围**（owner 已明确只做内部生成）；因此
  `hybrid_retrieval_selfaudit_DESIGN.md` 里"per-segment caption+embedding 索引"那条**不做**。

---

## 8. 与现有代码/文档的映射

| 设计要素 | 落点 |
|---|---|
| 读侧 fast/slow → 产出 IntentPlan | `skills/intent_understanding/interpreter.py`（§2.2） |
| 名字/别名匹配（定 `entity_id`） | `skills/memory_retrieval/name_match.py` |
| 语义候选召回（只出 candidates） | `skills/memory_retrieval/`（`Qwen3-Embedding-4B`，旁挂索引） |
| 确认门（semantic→身份敲定） | 新开上下文 MLLM 复核 / 人工（§4.1） |
| 双向匹配 / 身份锚定 | `skills/memory_update/curator.py`（`_resolve_asset` + `register_alias`） |
| write-through 物化 + 质量门 | `skills/memory_update/curator.py`（`MemoryUpdater`） |
| WeDetect-Ref 裁图 | `skills/crop_acquisition/`（替代 SAM3 概念分割） |
| 导出 memory.json + 旁挂 emb/帧 + JSON 图 | `skills/memory_update/snapshot.py` |
| deprecate 控制通道 | 人/规则写 `control.deprecate`；curator 标废弃留痕 |

---

## 9. 开放项 / 后续（待逐项确认再实现）

**已按两轮 gpt-5.5 评审修正（v1.0 定稿候选）**：拆 IntentPlanV1 / ResolvedPlanV1 两 schema；`semantic` 只出候选+确认门
（不越身份红线，ResolvedPlan 例已改为"确认通过"的自洽形态）；state 用闭集 `state_type`+自由 `state_description`、
`state.select` 判别式联合（全对象、含 `null`）；`identity_description` 与 `state_description` 分离；
`materialize` 意图层只给 `policy`（`grounding_query=null` 时默认用 `ref`）；`relations` 按 `key` 引用（`^t\d+$` 唯一）；
`create` 与懒物化区分；`write.expected_angles` 补回与 §4.3 对齐；per-target `status`+`failure_reason` 部分失败语义；
`deprecate` 移出 IntentPlan；cache 命中/未命中表述与确认门/懒读对齐。

待确认 / 后续：

1. **schema 冻结**：上述修正后的 IntentPlanV1（§2.2）是否冻结为 v1.0？
2. **guided decoding**（已定）：格式兜底用 `json_schema strict`；小工具模型待 owner 指认再替换。
3. **WeDetect-Ref 常驻服务**：接口/阈值（grounding score）与 crop 裁剪边距；`location` 走整帧不裁。
4. **确认门实现**：`needs_confirmation` 触发的新开上下文复核 prompt / 人工介入口。
5. **model_decided 选参考态**：`state_id` 枚举如何序列化给模型（temp0、留痕）。
6. **`relations` 的 compose 支持**：schema 已含，compose 侧成组逻辑实现期补（§4.5）。
7. **缓到 v1.1**：JSON 图的 **branch/retry**（`plan_revision`/`segment_status`，被拒重生成/分支编辑）；
   `relations` 是否从 SVO 收敛为 `interaction_group`。
8. **实现顺序**：接 `hybrid_retrieval_selfaudit_DESIGN.md` §6.5 的 T0–T3，本管线的 plan/执行器在 T0 之上落地。

> **不实现，直到 owner 逐项审定。** 本文件只负责把管线与 JSON 契约定清楚。
