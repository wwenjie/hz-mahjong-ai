"""自由匹配长跑：对战真实对手、记录战绩与战力（tasks.md 6A.1 / 6A.2 / 6A.5）。

一个会话的循环：

1. ``POST /api/match`` 取房号（等待期重复调用幂等返回原房，绝不双房双席）
2. 起 ``Runtime``，``tournament_id=房号``，跑到赛事进入终态或到达时长上限
3. **会话期间**周期采集事件流——**会话期间采不到**：自动房里「一局」指完整的 8 个回合，
   全部打完前一直是 ``403 GAME_NOT_FINISHED``；打完后窗口很短（房在约 60 秒后关停，
   连免认证端点一起 404）。所以采集必须足够密（默认 10 秒一次）才抓得住那个窗口
4. 记录战绩到 ``data/auto_sessions/sessions.jsonl``，并打印累计胜率与战力对比

为什么必须边打边采：实测 ``games/{batch}/events`` 只服务**最新一轮**，且未结束的局 403；
房一旦结束，``/api/test-rooms/{id}/games`` 也随之失效。所以漏掉那个窗口就永久丢失。

战力口径：用事件流里的 ``seats``（座位→user_id）定位我们的座位，再用 ``rounds[].scores``
取我们每一手的净分，因此可以算出**每手胜率、均番、总分**，并与同房另外三家逐一对比。

用法::

    uv run python tools/auto_session.py --sessions 6 --decider heuristic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# 允许直接以脚本方式运行（`uv run python tools/auto_session.py`）：把仓库根加进搜索路径，
# 否则下面的 `import tools.harvest_room` 会失败（脚本模式下 sys.path[0] 是 tools/ 而非仓库根）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from majiang.cli import make_decider
from majiang.client.api import PlatformApi
from majiang.client.transport import HttpTransport
from majiang.runtime.engine import Runtime, RuntimeOptions
from majiang.strategy.policy import Mode
from tools.harvest_room import harvest

DEFAULT_SERVER = "https://10.240.169.190:18080"
DEFAULT_OUT = "data/auto_sessions"
# 事件流只在**整场（8 回合）全部打完**后才可取（未打完一直是 403 GAME_NOT_FINISHED），
# 而房在打完约 60 秒后就关停、连免认证端点一起 404。因此采集必须足够密才抓得住那个窗口。
HARVEST_INTERVAL_SEC = 10.0
# 单会话时长上限。实测对手超时较多时一回合要 8 分钟量级，8 回合超过 1 小时，
# 所以上限给到 1 小时——太短会在打完之前就退出，既采不到数据也白费挂机。
DEFAULT_SESSION_CAP_SEC = 3600.0
MATCH_MIN_INTERVAL_SEC = 7.0  # 平台限速 10 次/分/用户


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def collect_stats(root: Path, user_id: str) -> dict:
    """从已采到的事件流里统计我们的战绩，并与同房三家对比。

    **必须遍历 ``rounds`` 的每一局**：一个事件流文件覆盖该场（8 回合）的**全部**回合，
    只取 ``rounds[0]`` 会把 8 局当成 1 局（曾因此把 6248 手算成 792 手）。
    """
    ours_score = 0
    wins = draws = hands = 0
    fan_total = 0
    per_user: dict[str, dict] = {}
    for path in sorted((root / "events").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seats = payload.get("seats") or []
        if len(seats) != 4:
            continue
        ids = [str(seat.get("user_id", "")) for seat in seats]
        mine = ids.index(user_id) if user_id in ids else None
        for result in payload.get("rounds") or []:
            scores = result.get("scores") or []
            if len(scores) != len(seats):
                continue
            hands += 1
            for index, who in enumerate(ids):
                bucket = per_user.setdefault(who, {"score": 0, "wins": 0, "hands": 0})
                bucket["score"] += int(scores[index])
                bucket["hands"] += 1
            if result.get("is_draw"):
                draws += 1
                continue
            winner = int(result.get("winner", -1))
            if 0 <= winner < len(ids):
                per_user[ids[winner]]["wins"] += 1
            if mine is None:
                continue
            ours_score += int(scores[mine])
            if winner == mine:
                wins += 1
                fan_total += int(result.get("multiplier", 1) or 1)
    return {
        "hands": hands,
        "our_score": ours_score,
        "our_wins": wins,
        "our_draws": draws,
        "our_win_rate": (wins / hands) if hands else 0.0,
        "our_average_fan": (fan_total / wins) if wins else 0.0,
        "per_user": per_user,
    }


def run_one_session(
    api: PlatformApi,
    transport: HttpTransport,
    user_id: str,
    *,
    decider_name: str,
    mode: Mode,
    options: RuntimeOptions,
    session_cap: float,
    out_root: Path,
    harvest_interval: float = HARVEST_INTERVAL_SEC,
) -> dict:
    match = api.match()
    room = match.room_id
    if not room:
        raise RuntimeError("POST /api/match 未返回房号")
    config = match.config
    started = now_iso()
    print(
        f"[{started}] 入席 {room} M={config.get('M')} Rounds={config.get('Rounds')} "
        f"出牌{config.get('DiscardTimeoutSec')}s 碰{config.get('PengTimeoutSec')}s "
        f"吃{config.get('ChiTimeoutSec')}s",
        flush=True,
    )

    room_root = out_root / room
    stop = threading.Event()
    harvest_state: dict = {"fresh": 0, "rounds": 0}

    def harvester() -> None:
        while not stop.is_set():
            try:
                _, fresh, _ = harvest(transport, room, room_root, force=False)
                harvest_state["fresh"] += fresh
                harvest_state["rounds"] += 1
            except Exception as exc:  # noqa: BLE001 —— 采集失败不能拖垮对局
                print(f"  采集异常（忽略）: {type(exc).__name__}: {exc}", flush=True)
            stop.wait(harvest_interval)

    thread = threading.Thread(target=harvester, name="harvest", daemon=True)
    thread.start()
    try:
        runtime = Runtime(
            api.token or "",
            server=transport.base_url,
            decider=make_decider(decider_name, mode),
            options=options,
            tournament_id=room,
        )
        summary = runtime.run(duration_sec=session_cap)
        runtime_fields = {
            "stop_reason": summary.stop_reason,
            "status": summary.status,
            "games_started": summary.games_started,
            "games_completed": summary.games_completed,
            "actions_submitted": summary.actions_submitted,
            "snapshots": summary.snapshots,
            "errors": summary.errors,
        }
    finally:
        stop.set()
        thread.join(timeout=HARVEST_INTERVAL_SEC + 5)

    # 收尾再采一次：会话刚结束时最后几局可能刚好完成
    try:
        harvest(transport, room, room_root, force=False)
    except Exception:  # noqa: BLE001
        pass

    stats = collect_stats(room_root, user_id)
    record = {
        "room_id": room,
        "started_at": started,
        "ended_at": now_iso(),
        "decider": decider_name,
        "mode": mode.value,
        "room_config": dict(config),
        "runtime": runtime_fields,
        "harvest": harvest_state,
        **stats,
    }
    record["our_rank"] = rank_of(stats, user_id)
    return record


def rank_of(stats: dict, user_id: str) -> int | None:
    per_user = stats.get("per_user") or {}
    if user_id not in per_user or len(per_user) < 2:
        return None
    ordered = sorted(per_user.items(), key=lambda kv: -kv[1]["score"])
    return next((i + 1 for i, (who, _) in enumerate(ordered) if who == user_id), None)


def print_record(record: dict, user_id: str) -> None:
    stats = record
    print(
        f"  战绩：{stats['hands']} 手  我们的分 {stats['our_score']:+d}  "
        f"胜 {stats['our_wins']} 流局 {stats['our_draws']}  "
        f"胜率 {stats['our_win_rate']:.1%}  均番 {stats['our_average_fan']:.2f}  "
        f"同房名次 {record.get('our_rank')}/{len(stats.get('per_user') or {})}",
        flush=True,
    )
    for who, bucket in sorted(
        (stats.get("per_user") or {}).items(), key=lambda kv: -kv[1]["score"]
    ):
        mark = " ← 我们" if who == user_id else ""
        print(f"    {who}: 分 {bucket['score']:+5d}  胡 {bucket['wins']:2d}/{bucket['hands']:2d}{mark}")
    runtime = record["runtime"]
    print(
        f"  运行：{runtime['stop_reason']}  开局 {runtime['games_started']}  "
        f"提交 {runtime['actions_submitted']}  快照 {runtime['snapshots']}  错误 {runtime['errors']}",
        flush=True,
    )
    print(f"  采集：新增 {record['harvest']['fresh']} 个事件流", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="自由匹配长跑会话")
    parser.add_argument("--sessions", type=int, default=1, help="跑几个会话（0 = 无限）")
    parser.add_argument(
        "--decider",
        default="heuristic",
        help="逗号分隔；多个值时**按会话轮换**——这是真机上的交错 A/B，能抵消对手组合与"
        "时间漂移（顺序对比做不到这一点）",
    )
    parser.add_argument("--mode", default="qualifier", choices=["qualifier", "final"])
    parser.add_argument("--rate", type=float, default=14.0)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--token-env", default="MATCH_TOKEN", help="存有全局令牌的环境变量名")
    parser.add_argument("--session-cap", type=float, default=DEFAULT_SESSION_CAP_SEC)
    parser.add_argument("--harvest-interval", type=float, default=HARVEST_INTERVAL_SEC)
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env, "")
    if not token:
        print(f"环境变量 {args.token_env} 里没有全局令牌", file=sys.stderr)
        return 1
    transport = HttpTransport(args.server)
    api = PlatformApi(transport, token)
    me = api.me()
    print(f"身份 {me.user_id}（全局令牌，tournament_id={me.tournament_id!r}）", flush=True)
    options = RuntimeOptions(rate_per_sec=args.rate, log_dir=args.log_dir, log_console=True)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ledger = out_root / "sessions.jsonl"

    mode = Mode(args.mode)
    deciders = [name.strip() for name in args.decider.split(",") if name.strip()]
    if not deciders:
        print("--decider 为空", file=sys.stderr)
        return 1
    print(f"决策器轮换表: {deciders}", flush=True)
    done = 0
    while args.sessions == 0 or done < args.sessions:
        # 交错而非顺序：同一时段内交替使用不同档位，抵消对手组合与时间漂移
        chosen = deciders[done % len(deciders)]
        try:
            record = run_one_session(
                api,
                transport,
                me.user_id,
                decider_name=chosen,
                mode=mode,
                options=options,
                session_cap=args.session_cap,
                out_root=out_root,
                harvest_interval=args.harvest_interval,
            )
        except Exception as exc:  # noqa: BLE001 —— 单会话失败不影响后续
            print(f"会话异常：{type(exc).__name__}: {exc}", flush=True)
            time.sleep(MATCH_MIN_INTERVAL_SEC * 2)
            continue
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{record['ended_at']}] 会话结束 {record['room_id']}", flush=True)
        print_record(record, me.user_id)
        done += 1
        time.sleep(MATCH_MIN_INTERVAL_SEC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
