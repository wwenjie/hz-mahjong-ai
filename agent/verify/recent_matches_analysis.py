"""近 3 天线上对局 vs 强 bot 分析（用户 19:18 任务②）。

回答两个问题：
1. 相比强 bot，我们输在哪（胡率/均番/得分结构/庄闲/巡目效率）
2. 还有没有其他之前没分析过的输分来源
"""
import json, glob, os, sys, random, statistics
sys.path.insert(0, "src")
from majiang.sim import replay as R
from collections import defaultdict

random.seed(7)
files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
# 近3天：按文件 mtime >= 2026-10-03 00:00
import datetime
cutoff = datetime.datetime(2026, 10, 3).timestamp()
recent = [f for f in files if os.path.getmtime(f) >= cutoff]
sample = random.sample(recent, min(2000, len(recent)))
print(f"近期房数: {len(recent)}，采样: {len(sample)}", flush=True)

# 我方 vs 对手聚合
we = defaultdict(float)     # 我方
opp = defaultdict(float)    # 对手（平均）
strong = defaultdict(float) # 强 bot（本场第一名）
n_games = 0
n_rounds = 0

# 结构维度
dealer_rounds = {"we": 0, "we_hu": 0, "opp": 0, "opp_hu": 0}
hu_by_fan = {"we": defaultdict(int), "opp": defaultdict(int)}
round_len_when_we_hu = []
round_len_when_opp_hu = []
self_draw = {"we": 0, "opp": 0}
rong = {"we": 0, "opp": 0}
deal_in = {"we": 0, "opp": 0}  # 点炮次数
baotou = {"we": 0, "opp": 0}
piao = {"we": 0, "opp": 0}

for fpath in sample:
    try:
        d = json.load(open(fpath))
    except Exception:
        continue
    seats = d.get("seats", [])
    our = next((i for i, s in enumerate(seats) if s.get("name") == "凤凰-5531"), None)
    if our is None:
        continue
    try:
        for state, events in R.iter_rounds(d):
            n_rounds += 1
            turn_count = 0
            winner = None; rscore = None; fan = 0; detail = []
            for ev in events:
                t = ev.get("type")
                if t == "tile_drawn" and ev.get("seat") == our:
                    turn_count += 1
                if t == "round_ended":
                    data = ev.get("data") or {}
                    rscore = data.get("scores")
                    fan = data.get("fan", 0)
                    detail = data.get("detail", [])
                    winner = ev.get("seat")
                    n_games += 1
                    if rscore and len(rscore) == 4:
                        we["score"] += rscore[our]
                        others = [rscore[i] for i in range(4) if i != our]
                        opp["score"] += sum(others) / 3
                        strong["score"] += max(rscore)
                        if rscore[our] == max(rscore):
                            we["rank1"] += 1
                    if winner is not None:
                        draw = data.get("draw", False)
                        if winner == our:
                            we["hu"] += 1
                            we["fan_sum"] += fan
                            hu_by_fan["we"][fan] += 1
                            round_len_when_we_hu.append(turn_count)
                            if draw:
                                self_draw["we"] += 1
                            else:
                                rong["we"] += 1
                            if "爆头" in str(detail): baotou["we"] += 1
                            if "财飘" in str(detail): piao["we"] += 1
                        else:
                            opp["hu"] += 1 / 3  # 每场一个对手胡
                            opp["fan_sum"] += fan / 3
                            hu_by_fan["opp"][fan] += 1
                            if not draw:
                                # 谁点的炮：scores 里负得最多的（非 winner）
                                if rscore:
                                    loser = min(range(4), key=lambda i: rscore[i])
                                    if loser == our:
                                        deal_in["we"] += 1
                                    else:
                                        deal_in["opp"] += 1/3
                            if "爆头" in str(detail): baotou["opp"] += 1/3
                            if "财飘" in str(detail): piao["opp"] += 1/3
                            if draw: self_draw["opp"] += 1/3
                            else: rong["opp"] += 1/3
                # 庄家维度
                if t == "round_ended":
                    pass
            # dealer 维度（本局庄）
            dealer = state.dealer if hasattr(state, 'dealer') else None
            if winner is not None:
                if dealer == our:
                    dealer_rounds["we"] += 1
                    if winner == our: dealer_rounds["we_hu"] += 1
                else:
                    dealer_rounds["opp"] += 1
                    if winner != our: dealer_rounds["opp_hu"] += 1
            for ev in events:
                R.apply_event(state, ev)
    except Exception:
        pass

print("=" * 70)
print(f"近3天线上对局再分析（{len(sample)} 房采样，{n_games} 局）")
print("=" * 70)
print()
print("【总量】")
print(f"  我方: 胡率 {we['hu']/n_games*100:.1f}%  均分 {we['score']/n_games:+.2f}  场首名率 {we['rank1']/max(n_games/8,1)*100:.1f}%")
print(f"  对手: 胡率 {opp['hu']/n_games*100:.1f}%  均分 {opp['score']/n_games:+.2f}")
print(f"  强bot(本场第一): 均分 {strong['score']/n_games:+.2f}")
print()
print("【胡牌质量】")
we_fan = we['fan_sum']/max(we['hu'],1)
opp_fan = opp['fan_sum']/max(opp['hu'],1)
print(f"  均番: 我方 {we_fan:.2f} vs 对手 {opp_fan:.2f} (差 {we_fan-opp_fan:+.2f})")
print(f"  自摸率(在胡牌中): 我方 {self_draw['we']/max(we['hu'],1)*100:.1f}% vs 对手 {self_draw['opp']/max(opp['hu'],1)*100:.1f}%")
print(f"  爆头: 我方 {baotou['we']:.0f}次 ({baotou['we']/n_games*100:.2f}%/局) vs 对手 {baotou['opp']:.0f} ({baotou['opp']/n_games*100:.2f}%)")
print(f"  财飘: 我方 {piao['we']:.0f}次 vs 对手 {piao['opp']:.0f}")
print()
print("【点炮/失分结构】")
print(f"  点炮: 我方 {deal_in['we']:.0f}次 ({deal_in['we']/n_games*100:.1f}%/局) vs 对手均 {deal_in['opp']:.1f}")
print()
print("【番型分布(胡牌次数, 前6档)】")
top_we = sorted(hu_by_fan['we'].items(), key=lambda x:-x[0])[:6]
top_opp = sorted(hu_by_fan['opp'].items(), key=lambda x:-x[0])[:6]
print(f"  我方: {top_we}")
print(f"  对手: {top_opp}")
print()
print("【效率】")
if round_len_when_we_hu:
    print(f"  我方胡牌时平均巡目: {statistics.mean(round_len_when_we_hu):.1f} (n={len(round_len_when_we_hu)})")
print()
print("【庄闲】")
print(f"  我方坐庄: {dealer_rounds['we']} 局, 胡 {dealer_rounds['we_hu']} ({dealer_rounds['we_hu']/max(dealer_rounds['we'],1)*100:.1f}%)")
print(f"  对手坐庄: {dealer_rounds['opp']} 局, 对手胡 {dealer_rounds['opp_hu']} ({dealer_rounds['opp_hu']/max(dealer_rounds['opp'],1)*100:.1f}%)")
print("ANALYSIS_DONE")
