---
name: mahjong-verify-metrics
description: 从原始事件流独立复算比赛指标（结构性不变量 → 指标 → 小样本手算对照），不 import A 的测量与回放代码。需要复核数字时使用。
---

# 独立复算

## 何时用

- A 报了一个数字，你要自己复算一遍。
- 数据量增长后要确认没有污染。
- 要判断某个结论建立在坏指标上。

## 命令

```bash
uv run python verify/invariants.py                       # 第一层：结构性不变量（每种牌全场 ≤4 张、总量 136、出牌必在手）
uv run python verify/metrics.py --manifest notes/manifest-20260926.txt
uv run python verify/spotcheck.py --n 10 --seed 42       # 第三层：小样本手算对照
uv run python verify/noise_floor.py --boot 10000         # 噪声底（复用，别自己发明）
uv run python verify/configure_check.py
uv run python verify/hand_progress.py --max-n 12
```

## 硬规则

1. **不 import**：`tools/measure_strength.py`、`tools/analyze_declined_wins.py`、`src/majiang/sim/replay.py`、
   以及 `verify/**` 里别人已写的复算脚本的**内部实现**（可以调用它的 CLI，不能复制它的解析逻辑）。
   理由：同一个 bug 会在两处犯同样的错。
2. **先不变量，后指标。** 结构性校验没过，后面所有数字作废。
3. **一份事件流文件 = 一场（8 局），不是一局。** 同一 `round_no` 可跨多个 block；只有每局首 block 带非空 `start_hands`。
   把 8 局铺在同一局面上算派生指标会全部作废——A 已经犯过一次。
4. **口径冲突要报告冲突本身**，附两个数值与各自口径，标为需人工裁定。不要自己选一个「更可信」的。
5. **记录快照指纹**（文件数与总 seq 范围），避免把数据增长误读为行为退化。
6. 杠的处理：`ming`/`bu`/`an` 都使 4 张离场入杠区，补牌从牌墙尾；**守恒校验只数「全场 ≤4」，不追踪墙序**，
   避免对平台摸牌顺序做假设。

## 不改别人地盘

`verify/**` 归 B。你要写自己的复算脚本请放 `agent/verify/`，只读 `data/` 与 `notes/`。

## 验收

- 不变量层通过；指标层给出与 A 口径的差异点（若有）。
- 小样本对照能手工核对通过。
