"""从已采的事件流里回填**对手名字**（用户 2026-09-30 00:58 要求，`majiang_rl` 线用作训练标签）。

**平台只给 AI 昵称、不给真名**（用户 11:49 已知悉）。昵称是稳定的身份标识：
`凤凰-5531` 是我们、`玄武-2346` 是榜首、`青龙/白虎/朱雀/玄武` 是平台内置 bot 家族。
对「按对手分层的训练与评测」够用——尤其 `majiang_rl` 那条线要按对手强度筛样本时，
只有昵称能把「同一个对手」的局聚起来。

产物两处（**刻意的双份**）：
- `data/auto_sessions/user_names.json`：完整映射 + 出现次数，机器本地，给 `majiang_rl` 直接读；
- `notes/opponent-names.tsv`：**入库**的紧凑表（`data/` 是 gitignore 的，不入库的东西别的线看不到
  ——这是 `notes/PROTOCOL.md` §2 的推论，我上一轮在这上面差点把证据放错地方）。

用法::

    uv run python tools/harvest_user_names.py
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

EVENTS = "data/auto_sessions/*/events/*.json"
OUT_JSON = Path("data/auto_sessions/user_names.json")
OUT_TSV = Path("notes/opponent-names.tsv")
OUR = "u_a7f7c67bb14a"


def main() -> int:
    names: dict[str, str] = {}
    appearances: collections.Counter = collections.Counter()
    rooms: dict[str, set] = collections.defaultdict(set)
    files = sorted(glob.glob(EVENTS))
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        seats = doc.get("seats") or []
        if len(seats) != 4:
            continue
        room = str(doc.get("room_id") or path.split("/")[-3])
        for seat in seats:
            who = str(seat.get("user_id", ""))
            name = str(seat.get("name", "") or "")
            if not who or not name:
                continue
            names.setdefault(who, name)
            appearances[who] += 1
            rooms[who].add(room)

    if not names:
        print("没有采到任何名字（事件流里没有 seats[].name）")
        return 1

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                who: {"name": name, "appearances": appearances[who], "rooms": len(rooms[who])}
                for who, name in sorted(names.items())
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    rows = sorted(names.items(), key=lambda kv: -appearances[kv[0]])
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 对手昵称表（平台只给 AI 昵称、不给真名）",
        "# 生成: tools/harvest_user_names.py；供 majiang_rl 线按对手分层训练/评测用",
        "# user_id\tname\t出现场次\t出现房数\t是否我们",
    ]
    for who, name in rows:
        lines.append(
            f"{who}\t{name}\t{appearances[who]}\t{len(rooms[who])}\t{'是' if who == OUR else ''}"
        )
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"扫描事件流 {len(files)} 个；识别出 {len(names)} 个单位（含我们）")
    print(f"写入 {OUT_JSON}（完整映射）与 {OUT_TSV}（入库紧凑表）")
    print("\n出现最多的 15 个：")
    for who, name in rows[:15]:
        mark = "  ← 我们" if who == OUR else ""
        print(f"  {name:28s} {appearances[who]:5d} 场 / {len(rooms[who]):4d} 房{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
