#!/usr/bin/env python3
"""「A vs B 对比必须先算 CI」守门探针（B' 2026-10-06 03:59 沉淀）。

今晚两次栽在「没算 CI 就报现象」：
1. 匹配后果对拍原始聚合 +4.5pp（分层匹配后降到 +1.51pp，CI 上限 <3pp）
2. 财神打出 vs 未打：对手 +1.42 ± 3.47 / 我方 −4.93 ± 15.09（两侧 CI 均含 0）

规则：任何「A 组比 B 组好/差 X」的结论，必须满足：
- 两组各自 n ≥ 30（否则标「样本不足，只报描述值不报方向」）
- 差值的 95% CI 不含 0（否则标「不显著」）
- 若是「匹配/分层」口径，必须报**分层加权后**的差值 + CI，不能只报原始聚合

用法（被其他分析脚本 import）：
    from tools.ci_gate import compare_two_groups
    verdict = compare_two_groups(group_a_scores, group_b_scores, label="财神 打出vs未打")
    print(verdict.report)   # 含 n / 均值差 / CI / 判定
    assert verdict.ok       # 不通过就不该下「A 比 B 好」的结论

也可直接命令行跑两组数（从文件读，一行一个数）：
    uv run python tools/ci_gate.py a.txt b.txt --label "..."
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field


@dataclass
class GroupVerdict:
    label: str
    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    diff: float | None          # mean_a - mean_b
    ci_lo: float | None
    ci_hi: float | None
    ok: bool                    # 是否足以报「有方向性差异」
    reasons: list[str] = field(default_factory=list)

    @property
    def report(self) -> str:
        def fmt(x, nd=3):
            return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "N/A"
        lines = [
            f"[{self.label}]",
            f"  n: A={self.n_a}, B={self.n_b}",
            f"  mean: A={fmt(self.mean_a)}, B={fmt(self.mean_b)}, diff(A-B)={fmt(self.diff)}",
            f"  95% CI of diff: [{fmt(self.ci_lo)}, {fmt(self.ci_hi)}]",
            f"  判定: {'✅ 可报方向性差异' if self.ok else '❌ 不可报方向（' + '; '.join(self.reasons) + '）'}",
        ]
        return "\n".join(lines)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _var(xs, m):
    if len(xs) < 2:
        return None
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def compare_two_groups(a: list[float], b: list[float], label: str = "compare",
                       min_n: int = 30) -> GroupVerdict:
    """独立双样本均值差 + 95% CI（Welch）。n<min_n 或 CI 含 0 ⇒ ok=False。"""
    v = GroupVerdict(label=label, n_a=len(a), n_b=len(b),
                     mean_a=None, mean_b=None, diff=None, ci_lo=None, ci_hi=None,
                     ok=False)
    if len(a) < 2 or len(b) < 2:
        v.reasons.append(f"样本太少（n_a={len(a)}, n_b={len(b)}，各需 ≥2）")
        return v
    ma, mb = _mean(a), _mean(b)
    v.mean_a, v.mean_b = ma, mb
    v.diff = ma - mb
    va, vb = _var(a, ma), _var(b, mb)
    se = math.sqrt(va / len(a) + vb / len(b))
    v.ci_lo = v.diff - 1.96 * se
    v.ci_hi = v.diff + 1.96 * se

    if len(a) < min_n or len(b) < min_n:
        v.reasons.append(f"样本不足（n_a={len(a)}, n_b={len(b)}，建议各 ≥{min_n}）；只报描述值")
        return v
    if v.ci_lo <= 0 <= v.ci_hi:
        v.reasons.append(f"95% CI [{v.ci_lo:+.3f}, {v.ci_hi:+.3f}] 含 0 ⇒ 不显著")
        return v
    v.ok = True
    return v


def _read_numbers(path: str) -> list[float]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(float(line))
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="A 组数值文件（一行一个）")
    ap.add_argument("b", help="B 组数值文件（一行一个）")
    ap.add_argument("--label", default="compare")
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()

    a = _read_numbers(args.a)
    b = _read_numbers(args.b)
    v = compare_two_groups(a, b, label=args.label, min_n=args.min_n)
    print(v.report)
    return 0 if v.ok else 1


if __name__ == "__main__":
    sys.exit(main())
