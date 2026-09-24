"""采集测试房历史对局事件流（tasks.md 6A.1）。

两个**免认证**端点（不需要令牌，因此可随时补采，不受比赛凭据影响）：

- ``GET /api/test-rooms/{id}/games`` —— 局列表 ``[{batch, game_id, round, status}]``
- ``GET /api/test-rooms/{id}/games/{batch}/events`` —— 该局完整事件流，**含四家起手手牌**
  （``blocks[].start_hands`` 为四家的 13/14 张牌码）、``round_ended.data`` 带**官方番数与
  番型标签**（``fan`` / ``detail``）、以及四家净分（``scores``）

### 实测得到的两个约束（不是推测）

1. **``{batch}`` 只服务最新一轮**。房间 ``t_0cfde5a00075`` 的列表有 60 条
   （``r{1..6}_b{0..9}_t0``），但 ``games/{batch}/events`` 的 batch=0..9 全部落在
   ``r6``，batch ≥ 10 返回 404。因此**可采的是最新一轮的 10 局，不是全部 60 局**。
   实践含义：要保住某一轮的原始事件流，必须在**该轮结束后尽快采**。
2. **采集按 batch 探测，不按列表回推**。``{batch}`` 参数与列表里的 ``batch`` 字段不是
   同一套编号，任何反推都可能错位；本工具直接迭代 batch 索引并信任响应里的 ``game_id``。

### 落盘策略

原始响应**原样保存**，解析与校验另做——避免解析逻辑有 bug 时把原始数据也丢掉。

用法::

    uv run python tools/harvest_room.py --room t_0cfde5a00075 --out data/harvest
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from majiang.client.errors import ApiError
from majiang.client.transport import HttpTransport
from majiang.rules.score import seat_deltas

DEFAULT_SERVER = "https://10.240.169.190:18080"
MAX_BATCH_PROBE = 200


def fetch_json(transport: HttpTransport, path: str):
    return transport.request("GET", path)


def discover(transport: HttpTransport, room: str) -> list[tuple[int, dict]]:
    """迭代 batch 索引直到连续 404，返回 ``[(batch, 事件流), ...]``。"""
    found: list[tuple[int, dict]] = []
    consecutive_miss = 0
    for batch in range(MAX_BATCH_PROBE):
        try:
            payload = fetch_json(transport, f"/api/test-rooms/{room}/games/{batch}/events")
        except ApiError as error:
            if error.status == 404:
                consecutive_miss += 1
                # 连续 5 次 404 视为越过末尾（单次空洞不应中断）
                if consecutive_miss >= 5:
                    break
                continue
            raise
        consecutive_miss = 0
        found.append((batch, payload))
    return found


def summarize(payload: dict, base_score: int) -> dict:
    """从一局的事件流里提取结论，并与我们的结算引擎对拍。

    **权威结果在顶层 ``rounds``**，不在事件流里。``blocks`` 只是按 seq 区间分页的
    事件分块（实测 b1 分成 [1,128] / [129,257] / [258,297] 三块），``round_ended``
    与 ``game_ended`` 只出现在**最后一块**——只看 ``blocks[0]`` 会误判成「该局未结束」。
    每一块都自带同一份 ``start_hands``（四家起手手牌），便于分块独立消费。
    """
    blocks = sorted(payload.get("blocks") or [], key=lambda b: b.get("seq_start", 0))
    events = [event for block in blocks for event in (block.get("events") or [])]
    rounds = payload.get("rounds") or []
    result = rounds[0] if rounds else {}
    ended = next((event for event in events if event.get("type") == "round_ended"), None)
    data = (ended or {}).get("data") or {}

    summary = {
        "game_id": payload.get("game_id"),
        "status": payload.get("status"),
        "blocks": len(blocks),
        "seq": [blocks[0].get("seq_start"), blocks[-1].get("seq_end")] if blocks else [],
        "start_hands": len(blocks[0].get("start_hands") or ()) if blocks else 0,
        "event_count": len(events),
        "event_types": dict(Counter(event.get("type") for event in events)),
        "dealer": result.get("dealer"),
        "round_no": result.get("round_no"),
        "is_draw": bool(result.get("is_draw")),
        "multiplier": result.get("multiplier"),
        "winner": result.get("winner"),
        "scores": result.get("scores"),
        "detail": data.get("detail"),
        "fan_from_event": data.get("fan"),
    }
    if not result:
        summary["ended"] = False
        return summary
    summary["ended"] = True
    scores = result.get("scores")
    if not scores:
        return summary
    if result.get("is_draw"):
        # 流局应当四家全零
        summary["draw_all_zero"] = all(value == 0 for value in scores)
        return summary
    expected = list(
        seat_deltas(
            fan=int(result["multiplier"]),
            base=base_score,
            winner_seat=int(result["winner"]),
            dealer_seat=int(result["dealer"]),
        )
    )
    summary["score_match"] = expected == list(scores)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="采集测试房历史对局事件流")
    parser.add_argument("--room", default="t_0cfde5a00075")
    parser.add_argument("--out", default="data/harvest")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--base-score", type=int, default=1, help="用于结算对拍的底分")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的原始文件")
    args = parser.parse_args(argv)

    transport = HttpTransport(args.server)
    root = Path(args.out) / args.room
    events_dir = root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    games = fetch_json(transport, f"/api/test-rooms/{args.room}/games").get("games", [])
    (root / "games.json").write_text(json.dumps(games, ensure_ascii=False, indent=1), encoding="utf-8")
    statuses = Counter(game.get("status") for game in games)
    rounds = Counter(game.get("round") for game in games)
    print(f"房间 {args.room}：列表 {len(games)} 局，状态 {dict(statuses)}，轮次 {dict(sorted(rounds.items()))}")

    streams = discover(transport, args.room)
    print(f"可采事件流 {len(streams)} 局（batch {[b for b, _ in streams]}）")

    stored = skipped = 0
    summaries = []
    for batch, payload in streams:
        path = events_dir / f"b{batch:03d}.json"
        if path.exists() and not args.force:
            skipped += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            stored += 1
        summaries.append(summarize(payload, args.base_score))

    (root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    ended = [s for s in summaries if s.get("ended")]
    wins = [s for s in ended if not s.get("is_draw")]
    draws = [s for s in ended if s.get("is_draw")]
    checked = [s for s in wins if "score_match" in s]
    matched = [s for s in checked if s["score_match"]]
    print(f"落盘 {stored} 个新文件，跳过 {skipped} 个已存在的")
    print(f"结论：可判定 {len(ended)} 局（胡牌 {len(wins)} / 流局 {len(draws)}）")
    if checked:
        print(f"结算对拍：{len(matched)}/{len(checked)} 与平台真实分数一致")
        bad = [s["game_id"] for s in checked if not s["score_match"]]
        if bad:
            print(f"  不一致（需要排查）: {bad}")
    bad_draws = [s["game_id"] for s in draws if s.get("draw_all_zero") is False]
    if bad_draws:
        print(f"  流局分非全零（异常）: {bad_draws}")
    if wins:
        print("番数分布:", dict(sorted(Counter(s["multiplier"] for s in wins).items())))
        labels = Counter(label for s in wins for label in (s.get("detail") or []))
        if labels:
            print("番型标签（取自 round_ended 事件）:", dict(labels))
        conflict = [
            s["game_id"]
            for s in wins
            if s.get("fan_from_event") is not None and s["fan_from_event"] != s["multiplier"]
        ]
        if conflict:
            print(f"  事件里的 fan 与顶层 multiplier 不一致: {conflict}")
    timeouts = sum(s.get("event_types", {}).get("timeout", 0) for s in summaries)
    total_events = sum(s.get("event_count", 0) for s in summaries)
    if total_events:
        print(
            f"超时事件 {timeouts} / 总事件 {total_events}（{timeouts / total_events:.0%}）"
            "—— 占比高说明这些局主要由服务端超时兜底，不是 AI 真实决策，"
            "只能用于校验规则与结算，不能用于评估策略"
        )
    unavailable = len(games) - len(streams)
    if unavailable > 0:
        print(
            f"覆盖度：本次只采到 {len(streams)} 局（最新一轮）；列表里的 {len(games)} 局中"
            f"还有 {unavailable} 局的事件流已不可取（实测 batch ≥ {len(streams)} 返回 404）。"
            "含义：要留住某一轮的原始事件流，必须在该轮结束后尽快采"
        )
    print(f"原始数据: {root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
