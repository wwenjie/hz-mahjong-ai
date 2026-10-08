---
name: mahjong-openspec-change
description: 在本仓库走 OpenSpec 流程建变更提案、写规格与设计、拆任务、实施、校验、归档。需要新增功能或改动行为时使用。
---

# OpenSpec 变更流程

## 何时用

- 新功能、行为改动、修复 —— 只要不是「改一行」，都先建变更。
- 需要看某个变更进行到哪一步。

## 硬前置：命令存在性校验

执行任何步骤前，先确认子命令存在；**不匹配就停下报告，不要猜命令**：

```bash
openspec --version
openspec --help | head -40
```

## 流程

```bash
openspec new change "<kebab-case-名>"        # 建变更（会生成 .openspec.yaml）
openspec status --change "<名>" --json        # 看 artifact 依赖顺序与 apply 门槛
openspec instructions <artifact> --change "<名>" --json   # 取模板与规则
openspec validate "<名>" --strict             # 校验
openspec list                                 # 列出在飞变更
openspec list --specs                         # 列出已有能力（写 Capabilities 前必看）
```

artifact 依赖顺序（spec-driven schema）：`proposal` → `specs` + `design` → `tasks`。

## 规则

1. **`context` 与 `rules` 是给你的约束，不是文件内容**，不要抄进 artifact。
2. Scenarios 必须用 **4 个 `#`**（`#### Scenarios: ...`），3 个会静默失效。
3. 每个 requirement 至少一个 scenario。
4. 新能力的 delta spec 第一段是 `## Purpose`（50 字以上）；改已有能力**不要**加 Purpose。
5. 每个任务必须是 `- [ ] X.Y 描述` 形式，且描述里写清验证方式（apply 阶段靠这个勾选）。

## 验收

- `openspec validate "<名>" --strict` 输出 valid。
- `openspec status --change "<名>"` 显示 `4/4 artifacts complete`。
