#!/usr/bin/env python3
"""副露同质性补充：按对手身份的 碰/吃接受率 × 胜率 关联（供 response_homogeneity 报告用）。"""
import collections
import glob
import json

import numpy as np

OUR = "u_a7f7c67bb14a"


def main():
    acc = collections.defaultdict(collections.Counter)
    wins = collections.Counter()
    rds = collections.Counter()
    name_of = {}
    for p in sorted(glob.glob("data/auto_sessions/*/events/*.json")):
        d = json.load(open(p))
        if d.get("status") != "finished":
            continue
        seats = d.get("seats") or []
        uids = [s.get("user_id") for s in seats]
        if OUR not in uids:
            continue
        our = uids.index(OUR)
        for s in range(4):
            if s != our:
                name_of[uids[s]] = seats[s].get("name", "?")
        for b in d["blocks"]:
            for e in b.get("events") or []:
                et, s, data = e["type"], e.get("seat"), e.get("data") or {}
                if et == "round_ended":
                    if data.get("draw"):
                        continue
                    for s2 in range(4):
                        rds[uids[s2]] += 1
                        if e.get("seat") == s2:
                            wins[uids[s2]] += 1
                    continue
                if s not in (0, 1, 2, 3):
                    continue
                u = uids[s]  # 含我们自己：ours 也进分布做对比
                if et in ("peng", "chi"):
                    acc[u][et + "_a"] += 1
                elif et == "gang" and data.get("kind") == "ming":
                    acc[u]["gang_a"] += 1
                elif et == "timeout" and data.get("kind") == "response":
                    acc[u][str(data.get("window")) + "_d"] += 1

    rows = []
    for u, c in acc.items():
        pw = c["peng_a"] + c["peng_d"]
        cw = c["chi_a"] + c["chi_d"]
        rows.append({
            "uid": u, "name": name_of.get(u, "?"), "rounds": rds[u],
            "win_rate": wins[u] / rds[u] if rds[u] else None,
            "peng_acc": c["peng_a"] / pw if pw >= 20 else None,
            "chi_acc": c["chi_a"] / cw if cw >= 20 else None,
            "melds": c["peng_a"] + c["chi_a"] + c["gang_a"],
        })
    rows.sort(key=lambda r: -(r["rounds"] or 0))
    print(f"{'name':<14}{'局':>6} {'胜率':>6} {'碰接受':>7} {'吃接受':>7} {'副露数':>6}")
    for r in rows[:15]:
        print(f"{r['name']:<14}{r['rounds']:>6} "
              f"{(f'{r[chr(34)+chr(34)]}' if False else format(r['win_rate'], '.1%')):>6} "
              f"{(format(r['peng_acc'], '.1%') if r['peng_acc'] is not None else '-'):>7} "
              f"{(format(r['chi_acc'], '.1%') if r['chi_acc'] is not None else '-'):>7} "
              f"{r['melds']:>6}")

    big = [r for r in rows if r["rounds"] >= 30 and r["peng_acc"] is not None
           and r["chi_acc"] is not None]
    wr = np.array([r["win_rate"] for r in big])
    pa = np.array([r["peng_acc"] for r in big])
    ca = np.array([r["chi_acc"] for r in big])
    print(f"\n身份级（≥30 局且窗口≥20）：n={len(big)}")
    print(f"corr(胜率, 碰接受率) = {np.corrcoef(wr, pa)[0,1]:+.2f}")
    print(f"corr(胜率, 吃接受率) = {np.corrcoef(wr, ca)[0,1]:+.2f}")
    print(f"胜率分布: 中位 {np.median(wr):.1%} IQR [{np.percentile(wr,25):.1%}, {np.percentile(wr,75):.1%}]")
    print(f"碰接受率: 中位 {np.median(pa):.1%} IQR [{np.percentile(pa,25):.1%}, {np.percentile(pa,75):.1%}]")
    print(f"吃接受率: 中位 {np.median(ca):.1%} IQR [{np.percentile(ca,25):.1%}, {np.percentile(ca,75):.1%}]")

    me = next((r for r in rows if r["uid"] == OUR), None)
    if me:
        def pctile(arr, v):
            return float((arr < v).mean())
        print(f"\n我们自己: 局={me['rounds']} 胜率={me['win_rate']:.1%} "
              f"碰接受={me['peng_acc']:.1%} 吃接受={me['chi_acc']:.1%}")
        print(f"分位（在 {len(big)} 个身份里）: 胜率超 {pctile(wr, me['win_rate']):.0%}、"
              f"碰接受超 {pctile(pa, me['peng_acc']):.0%}、吃接受超 {pctile(ca, me['chi_acc']):.0%}")


if __name__ == "__main__":
    main()
