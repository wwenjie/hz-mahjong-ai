#!/usr/bin/env python3
"""P2：总得分分布的尾部贡献（决赛该多激进的定量依据）。

口径：冻结清单全部 finished 场，取非流局 round_ended.data.scores。
- 每局赢家得分 = scores 中的最大值（零和，其余三家为负）
- 赢家得分按十分位分层，报各层对「赢家总得分」的贡献占比
- 分 us / opp 两组对比（我们胡的牌 vs 对手胡的牌，尾部厚度是否不同）
- 追加一个交易计算器：追番使胜率从 p 降到 p'、获胜局均番从 f 升到 f' 时，
  总得分的变化（在实测的分数-番数关系上估）

用法：uv run python verify/score_tail.py
"""
import collections
import json

OUR_UID = "u_a7f7c67bb14a"


def main():
    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    us_scores = []   # 我们赢的局的赢家得分
    opp_scores = []
    us_fan, opp_fan = [], []
    rounds = 0
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        seats = doc.get("seats") or []
        our = next((i for i, s in enumerate(seats) if s.get("user_id") == OUR_UID), None)
        if our is None:
            continue
        seen = set()
        for b in doc["blocks"]:
            for e in b.get("events") or []:
                if e["type"] != "round_ended":
                    continue
                rn = (e.get("data") or {}).get("round_no")
                if rn in seen:
                    continue
                seen.add(rn)
                d = e["data"]
                if d.get("draw"):
                    continue
                rounds += 1
                scores = d.get("scores") or [0, 0, 0, 0]
                w = e.get("seat")
                wscore = scores[w] if isinstance(w, int) and 0 <= w < 4 else max(scores)
                fan = d.get("fan") or 0
                (us_scores if w == our else opp_scores).append(wscore)
                (us_fan if w == our else opp_fan).append(fan)

    print(f"非流局={rounds}  我们赢={len(us_scores)}  对手赢={len(opp_scores)}")

    def tail_report(name, arr):
        arr = sorted(arr, reverse=True)
        total = sum(arr)
        n = len(arr)
        print(f"\n[{name}] n={n} 总得分={total} 均值={total/n:.2f}")
        cum = 0
        for q in (0.01, 0.05, 0.10, 0.25, 0.50):
            k = max(1, int(n * q))
            cum = sum(arr[:k])
            print(f"  前 {q:>4.0%}（{k:>4} 局，单局≥{arr[k-1]:>4} 分）贡献 {cum/total:6.1%}")
        print(f"  分位数: max={arr[0]} p99={arr[max(0,int(n*0.01)-1)]} "
              f"p90={arr[int(n*0.10)]} p50={arr[n//2]}")

    tail_report("我们胡的局", us_scores)
    tail_report("对手胡的局", opp_scores)

    def fan_stats(name, fans, scores):
        # 番→分的经验关系（用于交易计算器）
        by_fan = collections.defaultdict(list)
        for f, s in zip(fans, scores):
            by_fan[f].append(s)
        print(f"\n[{name}] 番→平均赢家得分:", {
            f: round(sum(v) / len(v), 1) for f, v in sorted(by_fan.items()) if len(v) >= 20})
    fan_stats("我们", us_fan, us_scores)
    fan_stats("对手", opp_fan, opp_scores)


if __name__ == "__main__":
    main()
