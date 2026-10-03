"""头部 bot vs 我们（凤凰-5531）行为差距分析。

从事件流直接算：胜率、均番、副露率、每手分。
数据源：data/auto_sessions/*/events/*.json
  - rounds[]: winner, scores, multiplier
  - blocks[].events[]: type=peng/chi/gang (副露), type=round_ended (fan, scores)
用法: setsid .venv/bin/python agent/verify/topbot_gap_analysis.py > agent/out/topbot-gap.txt 2>&1
"""
import json, glob, sys, os
from collections import Counter, defaultdict

TOP_BOTS = [
    "玄武-2346","三杯猫","铳一色14","歪比巴卜肉蛋葱鸡","爆头研究所",
    "Astra-0","腾蛇-0638","康陶应雀","凤凰-2626","glm-flash",
    "麒麟-7780","白虎-0211","豆包豆包帮我把其他AI电源拔掉",
    "Nomad","双白平胡","Kimi-K4.1","菜菜子","走马","今晚打老虎",
    "晴总总，该请桂语山房了",
]
OUR = "凤凰-5531"
MELD_TYPES = {"peng","chi","gang","minggang","angang","bugang"}

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    print(f"事件流: {len(files)}", flush=True)

    # Pass 1: name -> uid
    name_to_uid = {}
    for fpath in files:
        try:
            d = json.load(open(fpath))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception: pass
    missing = [n for n in TOP_BOTS if n not in name_to_uid]
    if missing: print(f"未命中: {missing}")
    targets = {n: name_to_uid[n] for n in TOP_BOTS + [OUR] if n in name_to_uid}
    uid_to_name = {v: k for k, v in targets.items()}
    print(f"目标玩家: {len(targets)}", flush=True)

    # Pass 2: 逐房统计
    stats = defaultdict(lambda: Counter())
    for i, fpath in enumerate(files):
        if i % 500 == 0: print(f"  {i}/{len(files)}", flush=True)
        try:
            d = json.load(open(fpath))
        except Exception: continue
        seats = d.get("seats", [])
        uid_seat = {}
        for si, s in enumerate(seats):
            u = s.get("user_id")
            if u in uid_to_name:
                uid_seat[u] = si
        if not uid_seat: continue

        # 轮级：胜负/番数/得分（从 round_ended 事件里取 fan 和 scores）
        for blk in d.get("blocks", []):
            for evt in blk.get("events", []):
                if evt.get("type") != "round_ended": continue
                rd = evt.get("data") or {}
                scores = rd.get("scores") or []
                fan = rd.get("fan", 0)
                winner = evt.get("seat")
                for uid, si in uid_seat.items():
                    st = stats[uid]
                    st["rounds"] += 1
                    if winner is not None and winner == si:
                        st["wins"] += 1
                        st["fan_sum"] += fan if isinstance(fan, (int, float)) else 0
                    if si < len(scores) and isinstance(scores[si], (int, float)):
                        st["score"] += scores[si]

            # 事件级：副露
            for evt in blk.get("events", []):
                seat = evt.get("seat")
                etype = evt.get("type", "")
                if etype not in MELD_TYPES: continue
                for uid, si in uid_seat.items():
                    if si == seat:
                        stats[uid]["melds"] += 1

    # 输出
    hdr = f"{'玩家':<24s} {'局数':>6s} {'胜率':>6s} {'均番':>6s} {'副露/局':>8s} {'每手分':>8s}"
    print(f"\n{hdr}")
    print("-" * 70)
    rows = []
    for uid, name in uid_to_name.items():
        st = stats[uid]
        r = st["rounds"]
        if r < 30: continue
        wr = st["wins"] / r * 100
        af = st["fan_sum"] / st["wins"] if st["wins"] else 0
        mr = st["melds"] / r
        pps = st["score"] / r
        rows.append((name, r, wr, af, mr, pps))
    rows.sort(key=lambda x: -x[5])
    for name, r, wr, af, mr, pps in rows:
        tag = " <<< 我们" if name == OUR else ""
        print(f"{name:<24s} {r:>6d} {wr:>5.1f}% {af:>6.3f} {mr:>8.3f} {pps:>+8.3f}{tag}")

    # 汇总
    our_row = [r for r in rows if r[0] == OUR]
    top_rows = [r for r in rows if r[0] != OUR]
    if our_row and top_rows:
        print(f"\n=== 头部 bot 均值 vs 我们 ===")
        tw = sum(r[1] for r in top_rows)
        for label, idx, fmt in [("胜率", 2, "{:.1f}%"), ("均番", 3, "{:.3f}"), ("副露/局", 4, "{:.3f}"), ("每手分", 5, "{:+.3f}")]:
            avg = sum(r[idx] * r[1] for r in top_rows) / tw
            ours = our_row[0][idx]
            diff = ours - avg
            print(f"  {label}: bot {fmt.format(avg)}  vs 我们 {fmt.format(ours)}  差 {fmt.format(diff)}")
    print("ANALYSIS_DONE", flush=True)

if __name__ == "__main__":
    main()
