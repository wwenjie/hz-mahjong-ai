#!/usr/bin/env python3
"""对拍门禁：跑一次配对 A/B，按「样本量 + 显著性」两道关决定候选能不能进策略。

存在理由（`notes/OWNERSHIP.md` 的教训）：A 曾据 6 场/臂、p≈0.10 的结果说「有 2.8 点优势」。
所以门禁把两件事做成**机械的**，而不是靠人记得：

1. **样本量关**：场数 < 200 时，不论统计量多好看，结论一律记「不显著（样本量不足）」
2. **显著性关**：`ab_test.py` 的 t 检验未过 |t|>1.96 时，记「无显著增益」

用法::

    uv run python research/compare.py --treatment risk --baseline heuristic --matches 200 --rounds 8

产物：``research/records/compare-<treatment>-vs-<baseline>.json``（含原始输出，可复查）。

噪声底口径见 ``verify/noise_floor.py``（按房 bootstrap，真机胜率差 SD ≈ 1.50%，检出 1.25pp 需 ~350 小时）。
本脚本做的是**自对弈配对对拍**，只用于筛掉明显无效的候选，**不能**据此判定「已到局部最优」。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "research", "records")
MIN_MATCHES = 200
LINE_RE = re.compile(
    r"^\s{2}(\S+)\s+均值\s+([+-][\d.]+)\s+标准误\s+([\d.]+)\s+t\s+([+-][\d.]+)"
    r"\s+95%CI\s+\[([+-][\d.]+),\s*([+-][\d.]+)\]\s+(\S+)")


def parse(text):
    out = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            out.append({"metric": m.group(1), "mean": float(m.group(2)),
                        "se": float(m.group(3)), "t": float(m.group(4)),
                        "ci_low": float(m.group(5)), "ci_high": float(m.group(6)),
                        "tool_verdict": m.group(7).strip("*")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--baseline", default="heuristic")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--metric", default="总得分", help="以此指标判定候选（默认总得分）")
    args = ap.parse_args()

    if args.matches < MIN_MATCHES:
        print(f"注意：场数 {args.matches} < 门槛 {MIN_MATCHES} —— 结论会记为「不显著（样本量不足）」，"
              f"不论统计量多好看。", file=sys.stderr)

    cmd = ["uv", "run", "python", "tools/ab_test.py",
           "--treatment", args.treatment, "--baseline", args.baseline,
           "--matches", str(args.matches), "--rounds", str(args.rounds),
           "--seed", str(args.seed)]
    print("运行：" + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=7200)
    text = proc.stdout + proc.stderr
    rows = parse(text)
    if proc.returncode != 0 or not rows:
        print("对拍失败或无法解析输出，退出码 "
              f"{proc.returncode}。原始输出尾部：\n{text[-800:]}", file=sys.stderr)
        sys.exit(1)

    picked = next((r for r in rows if r["metric"] == args.metric), rows[0])

    if args.matches < MIN_MATCHES:
        verdict, reason = "不显著（样本量不足）", f"{args.matches} 场/臂 < {MIN_MATCHES}"
    elif abs(picked["t"]) > 1.96:
        verdict = "有提升" if picked["mean"] > 0 else "有下降"
        reason = f"|t|={abs(picked['t']):.2f} > 1.96"
    else:
        verdict, reason = "无显著增益", f"|t|={abs(picked['t']):.2f} ≤ 1.96"

    record = {
        "compared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "treatment": args.treatment, "baseline": args.baseline,
        "matches": args.matches, "rounds": args.rounds, "seed": args.seed,
        "judged_metric": picked["metric"], "verdict": verdict, "reason": reason,
        "rows": rows,
        "gate": {"min_matches": MIN_MATCHES, "t_threshold": 1.96,
                 "noise_floor_tool": "verify/noise_floor.py"},
        "raw_tail": text[-2000:],
    }
    os.makedirs(RECORDS, exist_ok=True)
    out = os.path.join(RECORDS, f"compare-{args.treatment}-vs-{args.baseline}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n判定指标：{picked['metric']}")
    print(f"  均值 {picked['mean']:+.3f}  标准误 {picked['se']:.3f}  "
          f"t {picked['t']:+.2f}  95%CI [{picked['ci_low']:+.3f}, {picked['ci_high']:+.3f}]")
    print(f"  场数 {args.matches}/臂（门槛 {MIN_MATCHES}）  种子 {args.seed}")
    print(f"\n结论：{verdict}  依据：{reason}")
    if verdict != "有提升":
        print("默认决策器保持不变；模型产物归档备查。")
    print(f"记录：{out}")
    return 0 if verdict in ("有提升", "有下降") else 0


if __name__ == "__main__":
    sys.exit(main())
