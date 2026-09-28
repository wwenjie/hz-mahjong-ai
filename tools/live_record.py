"""v3 真机战绩 vs v2 时代（按房配对差分，只读台账）。

**为什么要按房配对**：不同房间的对手强度、座次、庄闲都不同，直接比绝对胡率会把
「房间难度」混进来。所以每家房算 **我们的胡率 − 同房三个对手的平均胡率**，
再对两个时代分别取均值——这条差分的房间间方差远小于绝对水平。

口径与限制（引用时必须写明）：
- 只读 `data/auto_sessions/sessions.jsonl` 里 `status=finished` 的会话。
- 对手侧的 win/hands 来自 `per_user`（较新会话才有）；老会话缺该字段则只能算绝对量。
- 「时代」以台账 `decider` 字段为准，v3 首个会话 started_at = 2026-09-28T14:44:43+08:00。
- 每房局数不同（M=10 或 8），故按「局」加权而不是按「房」平均。
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

LEDGER = Path("data/auto_sessions/sessions.jsonl")
OUR = "u_a7f7c67bb14a"


def load() -> list[dict]:
    rows = []
    for line in LEDGER.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if (doc.get("runtime") or {}).get("status") != "finished":
            continue
        rows.append(doc)
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    hands = wins = 0
    fan_weighted = 0.0
    ranks = Counter()
    scores = 0
    diff_samples: list[float] = []
    for doc in rows:
        per = doc.get("per_user") or {}
        mine = per.get(OUR) or {}
        h = int(mine.get("hands") or 0)
        w = int(mine.get("wins") or 0)
        if h:
            hands += h
            wins += w
            fan_weighted += float(doc.get("our_average_fan") or 0.0) * w
        others = [v for uid, v in per.items() if uid != OUR]
        oh = sum(int(v.get("hands") or 0) for v in others)
        ow = sum(int(v.get("wins") or 0) for v in others)
        if h and oh:
            diff_samples.append(w / h - ow / oh)
        if doc.get("our_rank"):
            ranks[int(doc["our_rank"])] += 1
        scores += int(doc.get("our_score") or 0)
    return {
        "name": name,
        "rounds": len(rows),
        "hands": hands,
        "wins": wins,
        "win_rate": wins / hands if hands else 0.0,
        "fan": fan_weighted / wins if wins else 0.0,
        "rank": ranks,
        "top1": (ranks[1] / sum(ranks.values())) if ranks else 0.0,
        "score_per_room": scores / len(rows) if rows else 0.0,
        "diff": statistics.mean(diff_samples) if diff_samples else float("nan"),
        "diff_n": len(diff_samples),
        "diff_se": (
            statistics.stdev(diff_samples) / len(diff_samples) ** 0.5
            if len(diff_samples) > 1
            else float("nan")
        ),
    }


def show(s: dict) -> None:
    print(f"\n== {s['name']}（{s['rounds']} 房 / {s['hands']} 手）==")
    print(f"  我们胡率 {s['win_rate']:.2%}（{s['wins']}/{s['hands']}）  均番 {s['fan']:.3f}")
    print(f"  每房平均得分 {s['score_per_room']:+.1f}   拿第 1 名占比 {s['top1']:.1%}"
          f"   名次分布 {dict(sorted(s['rank'].items()))}")
    if s["diff_n"]:
        print(f"  **按房配对差分（我们 − 同房对手均值）** {s['diff']:+.2%}"
              f"  ±{s['diff_se']:.2%}（n={s['diff_n']} 房）")


def main() -> int:
    rows = load()
    v2 = [d for d in rows if d.get("decider") == "v2"]
    v3 = [d for d in rows if d.get("decider") == "v3"]
    print(f"台账已结算 {len(rows)} 房：v2 {len(v2)} / v3 {len(v3)}"
          f" / 其它 {len(rows) - len(v2) - len(v3)}")
    show(summarize("v2 时代", v2))
    show(summarize("v3 时代", v3))
    print("\n读法：**看差分那一行**，绝对胡率会被房间难度污染；"
          "差分正 = 我们在该房里高于对手平均。v3 样本仍小，先看方向。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
