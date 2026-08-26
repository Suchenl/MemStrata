# 步骤 1.1：剧本设计 (Screenplay)

> **核心职责**：设计并冻结剧本与资产的 Schema 契约，彻底消除文学剧本与物理计划之间的断层。

---

## 🔑 核心机制 (Core Mechanisms)

1. **Human Layer First (人类创作层优先)**：
   - 整理出固定格式、可读、可审的 `human_readable_screenplay`。
   - 接近 Hollywood 剧本组织：slugline、action、character、dialogue、beats。
2. **Production Layer As Derived View (生产层派生视图)**：
   - 将人类剧本编译为可执行的 `production_screenplay`。
   - 包含 `scenes`、`shots`、`visual_track`、`audio_track`、`planned_assets`。
3. **PlannedAssetSlot (剧本 ↔ 计划双向映射器)**：
   - 承载剧本侧的“文学句柄”和计划侧的“物理槽位”。
   - 预留 `bound_asset_id` 槽位，当真实资产被提取或给定后，直接绑定到该槽位，消除信息断层。
4. **Bilingual Separated Dual-track (双语分离双轨制)**：
   - 中英文剧本分成两份独立文件（`*_en.json` 与 `*_zh.json`），使用完全相同的 ID 空间。
   - 英文版适合模型生成侧输入，中文版适合人类导演审阅。
