"""测试房联调：让多个身份到位保活，并抓取真实快照与事件流。

用途是把平台协议从「文档推断」变成「实测事实」：本脚本不提交任何动作（由服务端超时
兜底出牌），因此可以安全地用来观察真实的 ``/state`` 快照结构、事件形状、跨局行为与
阶段流转。

用法::

    set -a; . ./.env; set +a
    uv run python tools/capture_live.py --env-prefix TEST_TOKEN_ --minutes 25

令牌只从环境变量读取，不落盘到代码；抓取结果写入 ``data/live/<房间>/``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from majiang.client.api import PlatformApi
from majiang.client.errors import ApiError, TransportError
from majiang.client.transport import HttpTransport

DEFAULT_SERVER = "https://10.240.169.190:18080"
HEARTBEAT_INTERVAL = 20.0
POLL_INTERVAL = 5.0


@dataclass
class Session:
    name: str
    token: str
    api: PlatformApi
    user_id: str = ""
    tournament_id: str = ""
    last_heartbeat: float = 0.0
    last_ready_key: str = ""
    seen_games: set[str] = field(default_factory=set)
    last_seq: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    @property
    def label(self) -> str:
        return self.name


def collect_tokens(prefix: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for key, value in sorted(os.environ.items()):
        if key.startswith(prefix) and value.strip():
            found.append((key[len(prefix) :], value.strip()))
    return found


def dump(directory: Path, name: str, payload: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe(call, *args, **kwargs) -> Any:
    try:
        return call(*args, **kwargs)
    except (ApiError, TransportError) as exc:
        return exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="测试房联调与抓取")
    parser.add_argument("--server", default=os.environ.get("MAJIANG_SERVER", DEFAULT_SERVER))
    parser.add_argument("--env-prefix", default="TEST_TOKEN_")
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--out", default="data/live")
    args = parser.parse_args(argv)

    tokens = collect_tokens(args.env_prefix)
    if not tokens:
        print(f"环境变量里没有以 {args.env_prefix} 开头的令牌", file=sys.stderr)
        return 2

    transport = HttpTransport(args.server)
    sessions = [
        Session(name=name, token=token, api=PlatformApi(transport, token))
        for name, token in tokens
    ]

    room = ""
    for session in sessions:
        info = safe(session.api.me)
        if isinstance(info, Exception):
            print(f"[{session.label}] /api/me 失败: {info}", file=sys.stderr)
            continue
        session.user_id = info.user_id
        session.tournament_id = info.tournament_id
        room = room or session.tournament_id
        print(f"[{session.label}] user={info.user_id} room={info.tournament_id}")

    if not room:
        print("没有任何令牌绑定到锦标赛", file=sys.stderr)
        return 2

    out_dir = Path(args.out) / room
    deadline = time.monotonic() + args.minutes * 60
    print(f"\n房间 {room}，抓取目录 {out_dir}，运行 {args.minutes} 分钟\n")

    prepared = False
    while time.monotonic() < deadline:
        now = time.monotonic()

        for session in sessions:
            if now - session.last_heartbeat >= HEARTBEAT_INTERVAL:
                session.last_heartbeat = now
                state = safe(session.api.tournament, room)
                if isinstance(state, Exception):
                    session.errors += 1
                    print(f"[{session.label}] tournament 失败: {state}", file=sys.stderr)
                    continue
                if not prepared:
                    dump(
                        out_dir,
                        "tournament-before.json",
                        {
                            "status": state.status,
                            "registered_users": state.registered_users,
                            "ready_users": state.ready_users,
                            "my_games": list(state.my_games),
                            "my_games_by_batch": dict(state.my_games_by_batch),
                            "stage": None if state.stage is None else state.stage.name,
                        },
                    )
                if state.status in ("registering", "stage_open"):
                    key = f"{state.status}:{state.stage_status}"
                    if session.last_ready_key != key:
                        outcome = safe(session.api.ready, room)
                        if isinstance(outcome, Exception):
                            if not is_conflict(outcome):
                                print(f"[{session.label}] ready 失败: {outcome}", file=sys.stderr)
                        else:
                            session.last_ready_key = key
                            print(f"[{session.label}] 已到位（status={state.status}）")

            info = safe(session.api.me)
            if isinstance(info, Exception):
                session.errors += 1
                continue
            for game_id in info.active_games:
                if game_id in session.seen_games:
                    continue
                session.seen_games.add(game_id)
                envelope = safe(session.api.game_state, game_id, 0)
                if isinstance(envelope, Exception):
                    print(f"[{session.label}] {game_id} 快照失败: {envelope}", file=sys.stderr)
                    continue
                dump(out_dir, f"snapshot-{session.label}-{game_id}.json", envelope.raw)
                print(f"[{session.label}] 捕获快照 {game_id} seq={envelope.seq}")

            for game_id in list(session.seen_games):
                seq = session.last_seq.get(game_id, 0)
                envelope = safe(session.api.game_state, game_id, seq, timeout=30.0)
                if isinstance(envelope, Exception):
                    continue
                if envelope.events:
                    dump(
                        out_dir,
                        f"events-{session.label}-{game_id}-{envelope.seq}.json",
                        list(envelope.events),
                    )
                    print(
                        f"[{session.label}] {game_id} 事件 +{len(envelope.events)} -> seq={envelope.seq}"
                    )
                if envelope.snapshot is not None:
                    dump(out_dir, f"snapshot-{session.label}-{game_id}-{envelope.seq}.json", envelope.raw)
                    if envelope.gap:
                        print(f"[{session.label}] {game_id} 缺口快照 seq={envelope.seq}")
                session.last_seq[game_id] = envelope.seq

        prepared = True
        time.sleep(POLL_INTERVAL)

    print("\n抓取结束")
    return 0


def is_conflict(error: Exception) -> bool:
    return isinstance(error, ApiError) and error.status == 409


if __name__ == "__main__":
    sys.exit(main())
