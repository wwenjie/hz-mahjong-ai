---
name: mahjong-test-and-selfplay
description: 跑本仓库的测试、自对弈、校准与强度测量，并按噪声底纪律给结论。需要验证策略强弱或跑回归时使用。
---

# 测试与自对弈

## 何时用

- 改完代码要跑回归。
- 要比较两个决策器档位的强弱。
- 要按路线校准胜率与番数。

## 命令（照抄；括号内是默认值）

```bash
uv run pytest                                    # 全量测试
uv run pytest tests/test_x.py -k <name>          # 单测
uv run pytest tests/test_stability_*.py          # B 的稳定性演练（只跑，不改）

uv run python tools/selfplay.py --matches 30 --rounds 8 --seed 20260923
uv run python tools/ab_test.py --treatment <decider> --baseline heuristic \
    --matches 200 --rounds 8 --seed 20260923
uv run python tools/calibrate.py --commitment meld --rounds 1200 --seed 20260923
uv run python tools/measure_strength.py --events 'data/auto_sessions/*/events/*.json' \
    --min-hands 200 --by-decider
```

## 硬规则（每条都对应一次真实翻车）

1. **样本量不足就说「不显著」，不许给点估计式提升。**
   - 判据：任何「有提升」的断言至少 **200 场/臂**（对齐 `notes/STATUS.md` 里实验队列的既有口径）× **四座位旋转**；
     不足门槛时输出「不显著」，并给出实际场数。
   - 噪声底口径用 `verify/noise_floor.py`（按房 bootstrap），**不要自己发明统计方法**。
   - 仓库既有噪声底事实：真机胜率差按房 bootstrap SD ≈ 1.50%（二项的 6 倍），检出 1.25pp 需约 350 小时。
     所以「跑了几十场就看出优势」几乎必然是噪声。
2. **策略对比必须四座位旋转。** 把 treatment 固定坐 0 座，而 0 座就是庄家，会测出 +5.7 个百分点的假优势。
3. **抽样按房分层，不按文件。** 一个房四家固定，按文件抽样会把房间差异当策略差异。
4. **自对弈是有偏判据**，不能据此宣称「已到局部最优」。要强弱结论必须真机数据（那需要人批准）。
5. **报告必须带**：场数/局数、对比双方决策器档位、噪声底或置信区间、随机种子。
6. **跑长跑前先说要跑多久、占多少 CPU**，`nice -n 15` 起。

7. **sklearn/GBDT 上算力机跑多进程，必须写死单线程环境变量。**
   - 现象：worker CPU 满载但零产出、`load average` 远超核数、上下文切换几十万次——这是 OpenBLAS/MKL 线程超订
     （ProcessPool 每进程再开 N 个 BLAS 线程，608 线程抢 16 核）。
   - 修法：runner 脚本写死 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。
   - 验证：`grep Threads /proc/<pid>/status` 应显示 `Threads: 1`（不是 38）。
   - 实测：修复后单场从「不收敛」恢复到 ~10s，6 条任务从 12.8h 零产出恢复到 ~1h 出齐。

## 验收

- 输出里出现明确的场数与噪声底；结论段能区分「有提升」与「无显著增益」。
- 同种子重跑结论一致。
