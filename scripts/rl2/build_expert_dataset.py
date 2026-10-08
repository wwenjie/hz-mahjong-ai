#!/usr/bin/env python
"""从线上 auto_sessions 提取高手胜者决策点，构建 expert BC 数据集（D 线协作）。

数据源：majiang_ai/data/auto_sessions/
  - sessions.jsonl  ：房间元数据（per_user: uid → hands/wins/score）
  - user_names.json ：uid → 玩家名
  - {room}/events/{room}_r{R}_b{B}_t0.json ：全知事件流（起手牌 + 逐事件）

样本筛选：只保留「目标玩家（榜上前 N）赢的局」中该玩家的决策点。
timeout 过滤：timeout(kind=discard) 后同 seat 的 tile_discarded = 系统代打，剔除；
             timeout(kind=response) 后同 seat 的 pass = 系统代放弃，剔除。
  注：事件流无 hu 事件（平台能胡自动胡），hu 头无标签——BC 训练时 hu 头继续用
  现有 v7 数据或规则兜底。

输出：npz 分片（obs dict 各键 + y_action + y_tile），键与 nnrl2.obs.situation_to_obs 对齐。

用法::

    nice -n 19 python scripts/build_expert_dataset.py \
        --data-dir /home/wuwenjie01/majiang_ai/data/auto_sessions \
        --top-n 20 --metric avg_score \
        --out /home/wuwenjie01/majiang_rl2/data/expert_bc \
        --shard-size 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# 牌码解析（复用主仓）
sys.path.insert(0, "/home/wuwenjie01/majiang_ai/src")
from majiang.rules import tiles as tiles_mod  # noqa: E402

# ---- obs 常量（与 nnrl2.obs 对齐） ----
TILE_KINDS = 34
GOD = 33
MAX_DISCARDS = 24
MAX_MELDS = 4
MAX_HISTORY = 200
_MELD_KIND_TO_ID = {"chi": 0, "peng": 1, "gang": 2}

# ---- 动作标签（与 v7 对齐） ----
ACTION_DISCARD = 0
ACTION_CHI = 1
ACTION_PENG = 2
ACTION_GANG = 3
ACTION_HU = 4  # 无数据（平台自动胡）
ACTION_PASS = 5


def parse_tile(code: str) -> int:
    return tiles_mod.parse(code)


def build_leaderboard(data_dir: Path, metric: str, min_hands: int) -> list[tuple[str, str, float, int]]:
    """从 sessions.jsonl 聚合玩家战绩榜。

    Returns: [(uid, name, metric_value, hands), ...] 按 metric 降序
    """
    stats: dict[str, dict] = defaultdict(lambda: {"hands": 0, "wins": 0, "score": 0, "firsts": 0})
    with open(data_dir / "sessions.jsonl") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            per_user = rec.get("per_user") or {}
            if not per_user:
                continue
            # 首名 = 本房 score 最高者
            best = max(per_user.items(), key=lambda kv: kv[1].get("score", 0), default=None)
            for uid, u in per_user.items():
                s = stats[uid]
                s["hands"] += u.get("hands", 0)
                s["wins"] += u.get("wins", 0)
                s["score"] += u.get("score", 0)
                if best and best[0] == uid:
                    s["firsts"] += 1
    with open(data_dir / "user_names.json") as f:
        names = json.load(f)

    rows = []
    for uid, s in stats.items():
        if s["hands"] < min_hands:
            continue
        name = (names.get(uid) or {}).get("name") or uid
        if metric == "avg_score":
            val = s["score"] / s["hands"]
        elif metric == "win_rate":
            val = s["wins"] / s["hands"]
        elif metric == "first_rate":
            val = s["firsts"] / max(s["hands"] / 10, 1)  # hands≈局，10局≈1场
        else:
            raise ValueError(metric)
        rows.append((uid, name, val, s["hands"]))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


class RoundReplayer:
    """重放一个 block（一局）的事件流，维护四家状态，在目标玩家决策点产出样本。"""

    def __init__(self, block: dict, target_seats: set[int]):
        self.block = block
        self.target_seats = target_seats  # 目标玩家座位集合（本局赢家且上榜）
        # 状态
        start_hands = [h or [] for h in (block.get("start_hands") or [])]
        self.hands = [np.zeros(TILE_KINDS, dtype=np.int8) for _ in range(4)]
        for seat, hand_codes in enumerate(start_hands[:4]):
            for code in hand_codes:
                self.hands[seat][parse_tile(code)] += 1
        self.discards: list[list[int]] = [[], [], [], []]
        self.melds: list[list[tuple[int, int]]] = [[], [], [], []]  # (kind_id, tile)
        self.wall_remaining = 136 - sum(len(h) for h in start_hands)
        self.dealer = block["dealer"]
        self.last_discard: tuple[int, int] | None = None  # (seat, tile) 待响应的牌
        self.samples: list[dict] = []
        self.dropped_trivial = 0

    def _obs(self, seat: int, phase: int, target: int) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        obs["hand"] = self.hands[seat].copy()

        disc = np.full((4, MAX_DISCARDS), -1, dtype=np.int8)
        for i in range(4):
            rel = (i - seat) % 4
            for j, t in enumerate(self.discards[i][:MAX_DISCARDS]):
                disc[rel, j] = t
        obs["discards"] = disc

        melds = np.full((4, MAX_MELDS, 3), -1, dtype=np.int8)
        for i in range(4):
            rel = (i - seat) % 4
            for j, (kind_id, tile) in enumerate(self.melds[i][:MAX_MELDS]):
                melds[rel, j, 0] = kind_id
                melds[rel, j, 1] = tile
                melds[rel, j, 2] = 0
        obs["melds"] = melds

        hist: list[int] = []
        for i in range(4):
            hist.extend(self.discards[i])
        ah = np.full(MAX_HISTORY, -1, dtype=np.int8)
        if hist:
            ah[: len(hist)] = hist[:MAX_HISTORY]
        obs["action_history"] = ah

        obs["scores"] = np.zeros(4, dtype=np.int32)  # 局间累计分在外层补（v1 先置零）
        obs["god"] = np.int8(GOD)
        obs["wall_remaining"] = np.int32(self.wall_remaining)
        obs["turn"] = np.int8(sum(len(d) for d in self.discards))
        obs["phase"] = np.int8(phase)
        obs["target"] = np.int8(target)
        obs["seat"] = np.int8(seat)
        obs["dealer"] = np.int8(self.dealer)
        return obs

    def _avail_response(self, seat: int, target: int) -> np.ndarray:
        """响应阶段可行性（对齐 rules/action.py）：白板不可凑碰/杠/吃。"""
        avail = np.zeros(6, dtype=np.float32)
        avail[ACTION_PASS] = 1.0
        if target < 0 or target == GOD:
            return avail
        h = self.hands[seat]
        cnt = int(h[target])
        if cnt >= 2:
            avail[ACTION_PENG] = 1.0
        if cnt >= 3:
            avail[ACTION_GANG] = 1.0
        # chi: 三种搭子模式，只算字面搭子（百搭不可参与顺子）
        if target < 27:
            r = target % 9
            patterns = []
            if r >= 2:
                patterns.append((target - 2, target - 1))
            if 1 <= r <= 7:
                patterns.append((target - 1, target + 1))
            if r <= 6:
                patterns.append((target + 1, target + 2))
            for a, b in patterns:
                if h[a] >= 1 and h[b] >= 1:
                    avail[ACTION_CHI] = 1.0
                    break
        return avail

    def _record(self, seat: int, phase: int, target: int, y_action: int, y_tile: int):
        if seat not in self.target_seats:
            return
        if phase == 0:
            avail = np.zeros(6, dtype=np.float32)
            avail[ACTION_DISCARD] = 1.0
            if y_action == ACTION_GANG:
                # 补杠/暗杠：自己回合的决策，gang 可行
                avail[ACTION_GANG] = 1.0
        else:
            avail = self._avail_response(seat, target)
            # 丢掉「只有 pass 可行」的 trivial 样本：v7 训练对其零梯度，纯数据膨胀
            if y_action == ACTION_PASS and avail.sum() <= 1.0:
                self.dropped_trivial += 1
                return
        self.samples.append({
            "obs": self._obs(seat, phase, target),
            "y_action": y_action,
            "y_tile": y_tile,
            "avail": avail,
        })

    def replay(self):
        events = self.block["events"]
        n = len(events)
        for idx, e in enumerate(events):
            etype = e["type"]
            seat = e["seat"]
            if etype == "tile_drawn":
                t = parse_tile(e["tile"])
                self.hands[seat][t] += 1
                self.wall_remaining -= 1
            elif etype == "tile_discarded":
                # timeout 代打判定：前一事件是同 seat 的 timeout(kind=discard)
                prev = events[idx - 1] if idx > 0 else None
                auto = (
                    prev is not None
                    and prev["type"] == "timeout"
                    and prev["seat"] == seat
                    and (prev.get("data") or {}).get("kind") == "discard"
                )
                t = parse_tile(e["tile"])
                if not auto:
                    self._record(seat, phase=0, target=-1, y_action=ACTION_DISCARD, y_tile=t)
                if self.hands[seat][t] > 0:
                    self.hands[seat][t] -= 1
                self.discards[seat].append(t)
                self.last_discard = (seat, t)
            elif etype in ("chi", "peng", "gang"):
                # 事件自带 tile = 目标牌（响应窗口），比维护 last_discard 更可靠
                evt_tile = parse_tile(e["tile"]) if e.get("tile") else -1
                data = e.get("data") or {}
                gang_kind = data.get("kind", "") if etype == "gang" else ""
                if etype == "gang" and gang_kind in ("bu", "an", "ANGANG", "BUGANG"):
                    # 补杠/暗杠是自己回合的动作（非响应）：补杠手牌 -1，暗杠 -4
                    if gang_kind in ("an", "ANGANG"):
                        self.hands[seat][evt_tile] = max(0, self.hands[seat][evt_tile] - 4)
                        self.melds[seat].append((_MELD_KIND_TO_ID["gang"], evt_tile))
                    else:
                        self.hands[seat][evt_tile] = max(0, self.hands[seat][evt_tile] - 1)
                    # 补杠/暗杠样本：出牌阶段决策，phase=0 记为 gang 动作
                    self._record(seat, phase=0, target=-1, y_action=ACTION_GANG, y_tile=-1)
                    continue
                # 响应窗口的 chi/peng/明杠
                target_tile = evt_tile if evt_tile >= 0 else (self.last_discard[1] if self.last_discard else -1)
                self._record(seat, phase=1, target=target_tile,
                             y_action={"chi": ACTION_CHI, "peng": ACTION_PENG, "gang": ACTION_GANG}[etype],
                             y_tile=-1)
                kind_id = _MELD_KIND_TO_ID[etype]
                rep_tile = target_tile if target_tile >= 0 else 0
                self.melds[seat].append((kind_id, rep_tile))
                # 精确扣牌（物理过程）：chi/peng/gang = 拿进目标牌 + 从手牌出若干张成副露
                if etype == "chi":
                    # chi 的 data.tiles 是顺子 3 张（含目标牌）。手牌出「顺子中非目标的 2 张」。
                    chi_tiles = data.get("tiles") or []
                    for code in chi_tiles:
                        t = parse_tile(code)
                        if t != target_tile and self.hands[seat][t] > 0:
                            self.hands[seat][t] -= 1
                elif etype == "peng" and target_tile >= 0:
                    self.hands[seat][target_tile] = max(0, self.hands[seat][target_tile] - 2)
                elif etype == "gang" and target_tile >= 0:
                    self.hands[seat][target_tile] = max(0, self.hands[seat][target_tile] - 3)
                if self.last_discard is not None:
                    src_seat, t = self.last_discard
                    if self.discards[src_seat] and self.discards[src_seat][-1] == t:
                        self.discards[src_seat].pop()
                self.last_discard = None
            elif etype == "pass":
                prev = events[idx - 1] if idx > 0 else None
                auto = (
                    prev is not None
                    and prev["type"] == "timeout"
                    and prev["seat"] == seat
                    and (prev.get("data") or {}).get("kind") == "response"
                )
                if not auto and self.last_discard is not None:
                    self._record(seat, phase=1, target=self.last_discard[1],
                                 y_action=ACTION_PASS, y_tile=-1)
            elif etype in ("round_ended", "game_ended", "timeout"):
                pass
        return self.samples


def block_winner_seat(block: dict) -> tuple[int | None, tuple[int, ...]]:
    """从 round_ended 提取胜者座位与四家得分变化。"""
    for e in reversed(block["events"]):
        if e["type"] == "round_ended":
            data = e.get("data") or {}
            scores = tuple(data.get("scores", (0, 0, 0, 0)))
            if data.get("draw"):
                return None, scores
            return e["seat"], scores
    return None, (0, 0, 0, 0)


def shard_write(out_dir: Path, shard_idx: int, samples: list[dict]):
    if not samples:
        return
    obs_keys = samples[0]["obs"].keys()
    out = {}
    for k in obs_keys:
        out[k] = np.stack([s["obs"][k] for s in samples])
    out["y_action"] = np.array([s["y_action"] for s in samples], dtype=np.int64)
    out["y_tile"] = np.array([s["y_tile"] for s in samples], dtype=np.int64)
    out["avail"] = np.stack([s["avail"] for s in samples])  # (N, 6) float32
    path = out_dir / f"expert_bc.{shard_idx:05d}.npz"
    np.savez_compressed(path, **out)
    print(f"  shard {shard_idx}: {len(samples)} 样本 → {path}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/wuwenjie01/majiang_ai/data/auto_sessions")
    ap.add_argument("--out", default="/home/wuwenjie01/majiang_rl2/data/expert_bc")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--min-hands", type=int, default=20)
    ap.add_argument("--metric", default="avg_score", choices=["avg_score", "win_rate", "first_rate"])
    ap.add_argument("--uids-file", default="", help="外部指定 uid 名单（每行一个），优先于 top-n")
    ap.add_argument("--shard-size", type=int, default=50000)
    ap.add_argument("--max-rooms", type=int, default=0, help="调试用：只处理前 N 个房间")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 定目标 uid 集合
    if args.uids_file:
        target_uids = set(Path(args.uids_file).read_text().split())
        print(f"外部名单: {len(target_uids)} 个 uid", flush=True)
    else:
        board = build_leaderboard(data_dir, args.metric, args.min_hands)
        target_uids = {uid for uid, _, _, _ in board[: args.top_n]}
        print(f"榜单 ({args.metric}) top{args.top_n}:", flush=True)
        for i, (uid, name, val, hands) in enumerate(board[: args.top_n], 1):
            mark = "✓" if uid in target_uids else " "
            print(f"  {i:2d}. {name:<24} hands={hands:5d} {args.metric}={val:.3f} {mark}", flush=True)

    # 2. 遍历房间事件文件
    rooms = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("a_"))
    if args.max_rooms:
        rooms = rooms[: args.max_rooms]
    print(f"\n扫描 {len(rooms)} 个房间 ...", flush=True)

    shard: list[dict] = []
    shard_idx = 0
    total = 0
    kept_rounds = 0
    seen_rounds = 0
    for ri, room in enumerate(rooms):
        ev_dir = room / "events"
        if not ev_dir.is_dir():
            continue
        for fp in sorted(ev_dir.glob("*.json")):
            try:
                with open(fp) as f:
                    doc = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            seats = doc.get("seats") or []
            seat_uids = [s.get("user_id", "") for s in seats]
            for block in doc.get("blocks", []):
                seen_rounds += 1
                winner, _scores = block_winner_seat(block)
                if winner is None or winner >= len(seat_uids):
                    continue
                if seat_uids[winner] not in target_uids:
                    continue
                # start_hands 缺失或全空（0 张）则无法重建，跳过
                sh = block.get("start_hands")
                if not sh or sum(len(h or []) for h in sh) < 52:
                    continue
                kept_rounds += 1
                rep = RoundReplayer(block, target_seats={winner})
                shard.extend(rep.replay())
                if len(shard) >= args.shard_size:
                    shard_write(out_dir, shard_idx, shard)
                    total += len(shard)
                    shard = []
                    shard_idx += 1
        if (ri + 1) % 100 == 0:
            print(f"  [{ri+1}/{len(rooms)}] 局 {seen_rounds}→留 {kept_rounds}，样本累计 {total + len(shard)}", flush=True)

    if shard:
        shard_write(out_dir, shard_idx, shard)
        total += len(shard)

    meta = {
        "target_uids": sorted(target_uids),
        "metric": args.metric,
        "top_n": args.top_n,
        "rooms": len(rooms),
        "rounds_seen": seen_rounds,
        "rounds_kept": kept_rounds,
        "samples": total,
        "label_map": {"discard": 0, "chi": 1, "peng": 2, "gang": 3, "hu": 4, "pass": 5},
        "note": "hu 无事件（平台自动胡）；timeout 代打已剔除；scores 字段 v1 置零；avail=(N,6) 可行性 mask（响应点由手牌推导，出牌点仅 discard）",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n完成: {total} 样本，{kept_rounds}/{seen_rounds} 局命中，meta → {out_dir/'meta.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
