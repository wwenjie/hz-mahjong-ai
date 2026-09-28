"""赢牌画像：**别人赢在哪些番型上**（只读事件流，快）。

**为什么做**：`tools/analyze_opponents.py` 的天梯显示前 25 名对手胡率 25–32%/每手 +1.0~+3.7，
而我们冠军档约 20%/−1.0 —— 缺口有 +10pp 胡率那么大。胡率之外，**赢的番型结构**同样关键：
`round_ended.data.detail` 直接给出每一手赢了什么（`平胡` / `爆头` / `4个白板` / …），
`data.fan` 给出番。把这两样按 uid 聚合，就能回答「他们赢在哪些番型上」。

口径与限制：
- 只读**我们参与过的房**（对手因此都在「对我们」的场里被采样），控制房间条件。
- 分母用「该 uid 参与的局数」＝该房局数（每家都打满），所以胡率可直接比。
- `detail` 是番型标签列表；这里把每个标签的出现次数按 uid 统计，可得「赢的构成」。
- 事件流的 `rounds[]` 没有 detail，detail 只在 `blocks[].events[].round_ended.data` 里。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

OUR = "u_a7f7c67bb14a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 uid 的赢牌画像")
    parser.add_argument("--focus", nargs="*", default=[], help="要单独列出的 uid（可多个）")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    if args.limit:
        paths = paths[: args.limit]

    stats: dict[str, dict] = collections.defaultdict(
        lambda: {"rounds": 0, "wins": 0, "fan": 0, "labels": collections.Counter()}
    )
    for path in paths:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        # 局数：用 rounds[]（每局一条，含 is_draw/winner）
        rounds = doc.get("rounds") or []
        for _ in rounds:
            for uid in ids:
                stats[uid]["rounds"] += 1
        # 赢牌明细：扫事件流
        for block in doc.get("blocks") or []:
            for e in block.get("events") or []:
                if e.get("type") != "round_ended":
                    continue
                data = e.get("data") or {}
                seat = e.get("seat")
                if not isinstance(seat, int) or not 0 <= seat < 4:
                    continue
                if data.get("draw"):
                    continue
                uid = ids[seat]
                stats[uid]["wins"] += 1
                stats[uid]["fan"] += int(data.get("fan") or 0)
                for label in data.get("detail") or ():
                    stats[uid]["labels"][str(label)] += 1

    def line(uid: str) -> str:
        s = stats[uid]
        if not s["wins"]:
            return f"{uid:>16s} 无胡牌样本"
        baotou = s["labels"].get("爆头", 0)
        return (f"{uid:>16s} 局{s['rounds']:>5d} 胡{s['wins']:>4d} "
                f"胡率{s['wins'] / s['rounds']:>6.2%} 均番{s['fan'] / s['wins']:>5.2f} "
                f"爆头/胡{baotou / s['wins']:>6.1%} "
                f"4白板/胡{s['labels'].get('4个白板', 0) / s['wins']:>5.1%}")

    print(f"含我们的房文件 {len(paths)} 个\n")
    if args.focus:
        print("== 指定 uid ==")
        for uid in args.focus:
            if uid in stats:
                print(" " + line(uid))
        print()
    print("== 我们 ==")
    print(" " + line(OUR))
    print("\n== 全体按胡率排序（前 15，样本 ≥300 局）==")
    ranked = sorted(
        ((s["wins"] / s["rounds"], uid) for uid, s in stats.items()
         if s["rounds"] >= 300 and s["wins"]),
        reverse=True,
    )
    for rate, uid in ranked[:15]:
        print(" " + line(uid) + ("  ← 我们" if uid == OUR else ""))
    print("\n== 全体按均番排序（前 8，样本 ≥300 局、胡 ≥40）==")
    by_fan = sorted(
        ((s["fan"] / s["wins"], uid) for uid, s in stats.items()
         if s["rounds"] >= 300 and s["wins"] >= 40),
        reverse=True,
    )
    for fan, uid in by_fan[:8]:
        print(" " + line(uid) + ("  ← 我们" if uid == OUR else ""))
    if OUR in stats and stats[OUR]["wins"]:
        print("\n== 我们的番型构成（占胡牌比例）==")
        s = stats[OUR]
        for label, n in s["labels"].most_common(10):
            print(f"  {n:5d}  {n / s['wins']:>6.1%}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
