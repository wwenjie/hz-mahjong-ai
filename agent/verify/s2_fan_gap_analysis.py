"""S2 番型缺口对拍（A 21:25 分工表，coordinator 21:30 认领）。

任务：解释「我们均番 1.242 vs 头部 bot 1.422（−0.18）」的病根。
做法：全量事件流，对头部 20 bot vs 我们（凤凰-5531），按胡牌番型
（round_ended.data.detail[]）拆分次数与总 fan。

输出三组：
  1) 每番型：bot 次数/局、均 fan贡献  vs 我们 次数/局、均 fan贡献
     （每局番型率 = 该番型出现局数 / 总局数；每局 fan 贡献 = Σfan*局 / 总局数）
  2) fan 分布直方（1/2/3/4+）
  3) 缺口分解：总 fan/局 差 = Σ_番型 (bot率-bot率我们)*bot均fan + Σ (bot均fan-我们均fan)*我们率
     （率差项 vs 番数差项，看是「型少」还是「型同番低」）

数据源：data/auto_sessions/*/events/*.json
用法: setsid .venv/bin/python agent/verify/s2_fan_gap_analysis.py > agent/out/s2-fan-gap.txt 2>&1
"""
import json, glob, os
from collections import Counter, defaultdict

TOP_BOTS = [
    "玄武-2346","三杯猫","铳一色14","歪比巴卜肉蛋葱鸡","爆头研究所",
    "Astra-0","腾蛇-0638","康陶应雀","凤凰-2626","glm-flash",
    "麒麟-7780","白虎-0211","豆包豆包帮我把其他AI电源拔掉",
    "Nomad","双白平胡","Kimi-K4.1","菜菜子","走马","今晚打老虎",
    "晴总总，该请桂语山房了",
]
OUR = "凤凰-5531"

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
    targets = {n: name_to_uid[n] for n in TOP_BOTS + [OUR] if n in name_to_uid}
    uid_to_name = {v: k for k, v in targets.items()}
    print(f"目标玩家: {len(targets)}", flush=True)

    # Pass 2: 逐房累计
    # stats[uid]["rounds"]: 总局数（含流局/被胡）
    # stats[uid]["wins"]: 自胡局数
    # stats[uid]["type_rounds"][type]: 胡了且 detail 含该型的局数
    # stats[uid]["type_fan"][type]: 上述局的 fan 总和
    # stats[uid]["fan_dist"][fan]: 胡且 fan==fan 的局数
    stats = defaultdict(lambda: {"rounds": 0, "wins": 0,
                                 "type_rounds": Counter(), "type_fan": Counter(),
                                 "fan_dist": Counter()})

    for i, fpath in enumerate(files):
        if i % 1000 == 0: print(f"  {i}/{len(files)}", flush=True)
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

        for blk in d.get("blocks", []):
            for evt in blk.get("events", []):
                if evt.get("type") != "round_ended": continue
                rd = evt.get("data") or {}
                fan = rd.get("fan", 0)
                if not isinstance(fan, (int, float)): fan = 0
                detail = rd.get("detail") or []
                if not isinstance(detail, list): detail = [str(detail)]
                winner = evt.get("seat")
                for uid, si in uid_seat.items():
                    st = stats[uid]
                    st["rounds"] += 1
                    if winner is not None and winner == si:
                        st["wins"] += 1
                        st["fan_dist"][int(fan)] += 1
                        if detail:
                            for t in detail:
                                t = str(t)
                                st["type_rounds"][t] += 1
                                st["type_fan"][t] += fan
                        else:
                            st["type_rounds"]["<无detail>"] += 1
                            st["type_fan"]["<无detail>"] += fan

    # 汇总：top bots 加权（按局数）
    bot_uids = [u for u, n in uid_to_name.items() if n != OUR and stats[u]["rounds"] >= 30]
    our_uid = next((u for u, n in uid_to_name.items() if n == OUR), None)
    if not our_uid:
        print("未找到我们的 uid"); return
    total_bot_rounds = sum(stats[u]["rounds"] for u in bot_uids)
    our_rounds = stats[our_uid]["rounds"]
    print(f"\n头部 bot: {len(bot_uids)} 人 / 总 {total_bot_rounds} 局；我们 {our_rounds} 局")

    # 收集全部番型
    all_types = set()
    for u in bot_uids + [our_uid]:
        all_types.update(stats[u]["type_rounds"].keys())

    # 每番型表
    rows = []
    for t in all_types:
        b_tr = sum(stats[u]["type_rounds"][t] for u in bot_uids)
        b_tf = sum(stats[u]["type_fan"][t] for u in bot_uids)
        o_tr = stats[our_uid]["type_rounds"][t]
        o_tf = stats[our_uid]["type_fan"][t]
        b_rate = b_tr / total_bot_rounds if total_bot_rounds else 0
        o_rate = o_tr / our_rounds if our_rounds else 0
        b_avgfan = b_tf / b_tr if b_tr else 0
        o_avgfan = o_tf / o_tr if o_tr else 0
        # 每局 fan 贡献 = 局率 × 该型均 fan
        b_contrib = b_rate * b_avgfan
        o_contrib = o_rate * o_avgfan
        rows.append((t, b_tr, b_rate, b_avgfan, b_contrib,
                        o_tr, o_rate, o_avgfan, o_contrib))
    rows.sort(key=lambda r: -(r[4] + r[8]))  # 按双方总贡献排序

    print(f"\n=== 番型对照（按双方 fan/局贡献合计排序）===")
    hdr = (f"{'番型':<14s} | {'bot局数':>7s} {'bot率/局':>8s} {'bot均fan':>8s} {'bot fan/局':>10s}"
           f" | {'我局数':>6s} {'我率/局':>7s} {'我均fan':>7s} {'我 fan/局':>9s} | {'差(fan/局)':>11s}")
    print(hdr)
    print("-" * len(hdr))
    total_b_contrib = total_o_contrib = 0.0
    for t, b_tr, b_rate, b_af, b_c, o_tr, o_rate, o_af, o_c in rows:
        total_b_contrib += b_c; total_o_contrib += o_c
        print(f"{t:<14s} | {b_tr:>7d} {b_rate:>8.4f} {b_af:>8.3f} {b_c:>10.4f}"
              f" | {o_tr:>6d} {o_rate:>7.4f} {o_af:>7.3f} {o_c:>9.4f} | {o_c - b_c:>+11.4f}")
    print("-" * len(hdr))
    print(f"{'合计':<14s} | {'':>7s} {'':>8s} {'':>8s} {total_b_contrib:>10.4f}"
          f" | {'':>6s} {'':>7s} {'':>7s} {total_o_contrib:>9.4f} | {total_o_contrib - total_b_contrib:>+11.4f}")

    # fan 分布
    print(f"\n=== 胡牌 fan 分布（占各自胡局的比例）===")
    fan_keys = sorted({k for u in bot_uids + [our_uid] for k in stats[u]["fan_dist"]})
    bot_wins = sum(stats[u]["wins"] for u in bot_uids)
    our_wins = stats[our_uid]["wins"]
    print(f"{'fan':>4s} | {'bot局数':>8s} {'bot%':>7s} | {'我局数':>7s} {'我%':>7s} | {'pp差':>7s}")
    for fk in fan_keys:
        b_n = sum(stats[u]["fan_dist"][fk] for u in bot_uids)
        o_n = stats[our_uid]["fan_dist"][fk]
        b_p = b_n / bot_wins * 100 if bot_wins else 0
        o_p = o_n / our_wins * 100 if our_wins else 0
        print(f"{fk:>4d} | {b_n:>8d} {b_p:>6.2f}% | {o_n:>7d} {o_p:>6.2f}% | {o_p - b_p:>+6.2f}pp")

    # 缺口分解（rate-effect vs fan-effect，对全部番型）
    print(f"\n=== 缺口分解（对全部番型）===")
    rate_eff = fan_eff = cross = 0.0
    for t, b_tr, b_rate, b_af, b_c, o_tr, o_rate, o_af, o_c in rows:
        # Oaxaca 分解: b_c - o_c = (b_rate - o_rate) * b_af  [率差]
        #                        + o_rate * (b_af - o_af)  [番差]
        r_e = (b_rate - o_rate) * b_af
        f_e = o_rate * (b_af - o_af)
        rate_eff += r_e; fan_eff += f_e
    total_gap = total_b_contrib - total_o_contrib
    print(f"总 fan/局 缺口 = bot {total_b_contrib:.4f} − 我们 {total_o_contrib:.4f} = {total_gap:+.4f}")
    print(f"  率差项（bot 这种型打得多）     : {rate_eff:+.4f}")
    print(f"  番差项（同型 bot 番更高）       : {fan_eff:+.4f}")
    print(f"  残差（分解恒等式应≈0）          : {total_gap - rate_eff - fan_eff:+.6f}")
    print("ANALYSIS_DONE", flush=True)

if __name__ == "__main__":
    main()
