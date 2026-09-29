#!/usr/bin/env python
"""B1：独立复算 B' 的 unified 三臂 kill 判读（只读实验日志，零平台请求）。

做法（**不复用 B' 的任何代码**）：解析 `data/experiments/logs/unified*-vs-v3-*.log`
里每种子、每指标的 `均值 / 标准误 / t / 95%CI`，然后

1. 内部一致性：由 均值、t 反算 se 是否与日志 se 一致（±1e-3）；
   由 95%CI 反算 se 是否一致。
2. 合并 t（两种子）：按 B' 报的口径复算 pooled t = mean_pair / se_pair，
   其中 mean_pair=两种子均值之平均，se_pair=sqrt(se1²+se2²)/2；
   另用 Stouffer 法（t/√k 求和）作 **独立第二口径** 对读。
3. 同向性：两种子均值符号是否一致。
4. 与 B' 22:58 表里的「合并 t」对读，差异 >0.05 即标记。

若 B' 的判读与我的复算不符，按判据标出；相符则明写「复算确认」。
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
LOGS = REPO / "data" / "experiments" / "logs"

ARMS = {
    "unified": ("unified-vs-v3-s20260927-seed20260927.log",
                "unified-vs-v3-s771014-seed771014.log"),
    "unified-pure": ("unified-pure-vs-v3-s20260927-seed20260927.log",
                     "unified-pure-vs-v3-s771014-seed771014.log"),
    "unified-recal": ("unified-recal-vs-v3-s20260927-seed20260927.log",
                      "unified-recal-vs-v3-s771014-seed771014.log"),
}
# B' 22:58 表里报的「合并 t（名次分）」
B_PRIME_COMBINED_T = {"unified": -0.33, "unified-pure": -4.67, "unified-recal": -4.48}

METRICS = ("总得分", "名次分", "白板数", "胡次数", "番数总和")
LINE = re.compile(
    r"^(总得分|名次分|白板数|胡次数|番数总和)\s+均值\s+([+-]?[\d.]+)\s+标准误\s+([\d.]+)\s+"
    r"t\s+([+-]?[\d.]+)\s+95%CI\s+\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]"
)


def parse(path: pathlib.Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line.strip())
        if m:
            out[m.group(1)] = {
                "mean": float(m.group(2)),
                "se": float(m.group(3)),
                "t": float(m.group(4)),
                "ci_lo": float(m.group(5)),
                "ci_hi": float(m.group(6)),
            }
    return out


def main() -> int:
    print("=" * 78)
    print("B1 独立复算：unified 三臂 kill 判读（素材=data/experiments/logs/*.log）")
    print("=" * 78)
    problems = 0
    for arm, files in ARMS.items():
        per_seed = []
        for fn in files:
            p = LOGS / fn
            if not p.exists():
                print(f"[缺失] {p}")
                problems += 1
                continue
            per_seed.append((fn, parse(p)))
        print(f"\n── {arm} ──")
        for fn, d in per_seed:
            tag = fn.split("-")[-1].replace(".log", "")
            for metric in METRICS:
                if metric not in d:
                    continue
                r = d[metric]
                se_from_t = abs(r["mean"] / r["t"]) if r["t"] else float("nan")
                se_from_ci = (r["ci_hi"] - r["ci_lo"]) / (2 * 1.959964)
                # 容差用**相对**误差：mean/t 均被四舍五入到 2–3 位有效数字，
                # 对 se≈1.0 的指标，绝对容差 2e-3 会被舍入误判为“不一致”。
                tol = max(0.02, 0.02 * r["se"])
                # t 只印 2 位小数；|t| 很小时（如 0.14）它的舍入会主导 imputed se。
                # 加上 t 舍入传播项：|mean|·(0.005/|t|²)。
                if r["t"]:
                    tol = max(tol, abs(r["mean"]) * 0.005 / (r["t"] ** 2))
                ok_t = abs(se_from_t - r["se"]) <= tol
                ok_ci = abs(se_from_ci - r["se"]) <= tol
                flag = "" if (ok_t and ok_ci) else "  <-- 内部不一致!"
                if flag:
                    problems += 1
                print(f"  [{tag}] {metric:4s} mean={r['mean']:+.3f} se={r['se']:.3f} "
                      f"t={r['t']:+.2f} | se(t)={se_from_t:.3f} se(CI)={se_from_ci:.3f}{flag}")
        # 合并口径（两种子），只对 名次分 / 总得分 报
        for metric in ("名次分", "总得分"):
            if len(per_seed) < 2:
                continue
            (_, d1), (_, d2) = per_seed[0], per_seed[1]
            if metric not in d1 or metric not in d2:
                continue
            m = (d1[metric]["mean"] + d2[metric]["mean"]) / 2
            se = ((d1[metric]["se"] ** 2 + d2[metric]["se"] ** 2) ** 0.5) / 2
            t_pool = m / se if se else float("nan")
            t_stouffer = (d1[metric]["t"] + d2[metric]["t"]) / (2 ** 0.5)
            same_dir = (d1[metric]["mean"] >= 0) == (d2[metric]["mean"] >= 0)
            print(f"  合并[{metric}] pooled_t={t_pool:+.2f}  stouffer_t={t_stouffer:+.2f}  "
                  f"同向={same_dir}")
            if metric == "名次分":
                b = B_PRIME_COMBINED_T.get(arm)
                diff = abs(t_pool - b) if b is not None else float("nan")
                verdict = "复算确认" if diff <= 0.05 else f"**与 B' 不符（差 {diff:.2f}）**"
                print(f"    vs B' 报的合并 t = {b:+.2f}  → {verdict}")
                if diff > 0.05:
                    problems += 1
        # 判读复述
        m1 = per_seed[0][1]["名次分"]["mean"] if len(per_seed) else float("nan")
        m2 = per_seed[1][1]["名次分"]["mean"] if len(per_seed) > 1 else float("nan")
        same = (m1 >= 0) == (m2 >= 0)
        print(f"  判读核对：两种子名次分 mean = {m1:+.3f} / {m2:+.3f} → 同向={same} "
              f"⇒ B' 的 'kill' 结论{'不' if not same else ''}依赖同向性")
    print("\n" + "=" * 78)
    print(f"内部一致性问题数 = {problems}")
    print("结论：B' 的合并 t 可由日志的 mean/se 复算复现（pooled 口径）⇒ 判读算术成立。"
          if problems == 0 else "存在待解释的不一致，见上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
