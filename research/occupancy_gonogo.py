#!/usr/bin/env python3
"""agent-c · 占用预测的离线 Go/No-Go（用户 14:45 授权）。

**问题**：现有口径把「我们自己还摸得到几张」算成 `Σ max(0, 4 − 已见)`，
但**对手暗手里的牌我们永远摸不到**（本变体只能自摸）。实测：这个口径把可用听口
高估 **1.8×**，且计入对手暗手会改变 **27.4%** 的听牌出牌（60 房）。

**学习目标（34 维）**：`opp[w]` = 三个对手暗手里该牌种共几张（0..4）。
**判据（预登记）**：只用**公开信息**学 `opp_hat`，用它算
`T_hat = Σ max(0, 未现[w] − opp_hat[w])`；在**留出房**上，
`argmax(T_hat)` 与全信息 `argmax(T)`（其中 `T = Σ max(0, 未现[w] − opp[w])`）的一致率，
必须显著高于现有口径基线 **72.6%**，且高于手写行为加权 **67.7%**。

**合规**：标签 = 对手暗手（**仅离线**）；特征只用公开信息（自己暗手 + 各家弃牌 +
各家副露 + 牌墙 + 本人明面）。线上推理不读对手暗手；模型上线须走 `strategy/gbdt.py` 纯 Python 导出。

用法::

    nice -n 19 uv run python research/occupancy_gonogo.py --train-rooms 90 --test-rooms 60
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np

from majiang.rules import shanten as sm
from majiang.rules import tiles, win
from majiang.rules.situation import PHASE_DRAW
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"
CPK = tiles.COPIES_PER_KIND
TK = tiles.TILE_KINDS
SEATS = 4
SUITS = 4


def suit_of(t: int) -> int:
    return tiles.suit(t) if tiles.is_number(t) else 3


def tile_features(ctx: dict, w: int, mode: str = "base") -> list[float]:
    """某牌种 `w` 在该点上的公开特征。mode="recency" 追加弃牌顺序/时间衰减维度。"""
    seen = ctx["seen"]
    own = ctx["own"]
    s = tiles.is_number(w)
    left = seen[w - 1] if (s and w % 9 != 0) else -1
    right = seen[w + 1] if (s and w % 9 != 8) else -1
    od_total = ctx["od_total"]
    od = [v / od_total for v in ctx["od_suit"]] if od_total else [0.0] * SUITS
    base = [
        float(own[w]),
        float(seen[w]) / CPK,
        float(ctx["opp_disc"][w]),
        float(ctx["meld_with"][w]),
        float(left) / CPK,
        float(right) / CPK,
        1.0 if s else 0.0,
        float(suit_of(w) == 0),
        float(suit_of(w) == 1),
        float(suit_of(w) == 2),
        (tiles.rank(w) / 9.0) if s else 0.0,
        0.0 if s else 1.0,
        od[0], od[1], od[2], od[3],
        float(ctx["H"]) / 13.0,
        float(ctx["wall"]) / 84.0,
        float(ctx["progress"]),
        float(ctx["draws"]) / 84.0,
        float(ctx["n_opp_melds"]),
        float(ctx["n_opp_disc"]),
        float(ctx["gods_seen"]) / CPK,
    ]
    if mode != "recency":
        return base
    # ── 用户 18:23 的假设：弃牌顺序次关键、越久的越不重要 ──
    gd = ctx.get("gdisc") or []
    mine = ctx.get("mine", -1)
    L = len(gd)
    lam = 0.85
    decayed = 0.0
    last_pos = -1
    recent4 = 0
    for i, (sd, td) in enumerate(gd):
        if sd == mine:
            continue
        if td == w:
            decayed += lam ** (L - 1 - i)
            last_pos = i
        if i >= L - 4 and td == w:
            recent4 += 1
    since = (L - last_pos) if last_pos >= 0 else 0
    base += [
        decayed,
        float(since) / 12.0,
        float(recent4) / 4.0,
    ]
    return base


def build_ctx(sit, state, mine: int, counts14: list[int], global_disc=None) -> dict:
    seen = sm.visible_counts(counts14, [m.tiles for m in sit.all_melds], sit.discards)
    opp = [0] * TK
    H = 0
    for s in range(SEATS):
        if s == mine:
            continue
        for i, amount in enumerate(state.seats[s].hand):
            opp[i] += amount
        H += sum(state.seats[s].hand)
    opp_disc = [0] * TK
    od_suit = [0] * SUITS
    n_opp_disc = 0
    n_opp_melds = 0
    for s in range(SEATS):
        if s == mine:
            continue
        for d in state.seats[s].discards:
            opp_disc[d] += 1
            od_suit[suit_of(d)] += 1
            n_opp_disc += 1
        n_opp_melds += len(state.seats[s].melds)
    meld_with = [0] * TK
    for meld in sit.all_melds:
        for t in meld.tiles:
            meld_with[t] += 1
    gods_seen = sum(1 for g in sit.discards for t in g if t == tiles.GOD)
    gods_seen += sum(1 for meld in sit.all_melds for t in meld.tiles if t == tiles.GOD)
    return {
        "seen": seen, "own": counts14, "opp_disc": opp_disc, "meld_with": meld_with,
        "od_suit": od_suit, "od_total": n_opp_disc, "H": H,
        "wall": sit.table.wall_remaining, "progress": sit.table.progress,
        "draws": sit.table.draws_made, "n_opp_melds": n_opp_melds,
        "n_opp_disc": n_opp_disc, "gods_seen": gods_seen, "opp": opp,
        "gdisc": global_disc or [], "mine": mine,
    }


def scan(rooms: int, cap: int, seed: int, mode: str = "base"):
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    rng = random.Random(seed)
    rng.shuffle(files)
    files = files[:rooms]
    X: list[list[float]] = []
    Y: list[float] = []
    Ti: list[int] = []
    eval_ctxs: list[tuple[dict, dict[int, tuple[int, ...]]]] = []
    used = 0
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        used += 1
        mine = ids.index(OUR)
        taken = 0
        gdisc: list[tuple[int, int]] = []
        for event, state in replay.iter_before_each_event(doc):
            if event.get("type") != replay.DISCARDED or not state.opened:
                continue
            if (event.get("data") or {}).get("catch_play"):
                continue
            s_disc = event.get("seat")
            if s_disc is None or not (0 <= int(s_disc) < SEATS):
                continue
            ld = getattr(state, "last_discarder", None)
            td = getattr(state, "last_discard", None)
            if ld is not None and td is not None:
                gdisc.append((int(ld), int(td)))
            if taken >= cap:
                # 训练样本封顶，但**评估点不封顶**（否则只剩前 50 次出牌、样本过少）
                if int(s_disc) != mine:
                    continue
            try:
                sit = state.situation_for(mine, phase=PHASE_DRAW)
            except Exception:
                continue
            counts14 = list(sit.hand.counts)
            ctx = build_ctx(sit, state, mine, counts14, gdisc)
            if taken < cap:
                for w in range(TK):
                    X.append(tile_features(ctx, w, mode))
                    Y.append(float(ctx["opp"][w]))
                    Ti.append(w)
                taken += 1
            if int(s_disc) != mine:
                continue
            waitsets: dict[int, tuple[int, ...]] = {}
            for tile in range(TK):
                if counts14[tile] <= 0:
                    continue
                after = list(counts14)
                after[tile] -= 1
                try:
                    waits = win.winning_draws(after, sit.hand.meld_count)
                except ValueError:
                    continue
                if waits:
                    waitsets[tile] = waits
            if len({w for w in waitsets.values()}) >= 2:
                eval_ctxs.append((ctx, waitsets))
    return (
        np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64),
        eval_ctxs, used, np.asarray(Ti, dtype=np.int64),
    )


def score(eval_ctxs, predict_fn, seen_lookup=None, detail=None, mode: str = "base") -> tuple[int, ...]:
    """返回 (n, 一致_hat, 一致_seen, 一致_V, 一致_prop, 一致_beh)。

    一致_seen：若给出 `seen_lookup`（长度 CPK+1，`E[opp|seen=w]` 的训练集查表），
    用它作为「只复制已见张数」的基线——用于判定 34 维模型是否只是复制了 `seen`。

    `detail` 若传一个 dict，会把逐点布尔写入 `detail['hat']`/`['v']` 等（用于配对检验）。
    """
    n = same_hat = same_seen = same_v = same_prop = same_beh = 0
    if detail is not None:
        for k in ("hat", "seen", "v", "prop", "beh"):
            detail.setdefault(k, [])
    for ctx, waitsets in eval_ctxs:
        seen = ctx["seen"]
        opp = ctx["opp"]
        H = ctx["H"]
        od_total = ctx["od_total"]
        od = [v / od_total for v in ctx["od_suit"]] if od_total else [0.0] * SUITS
        sw = [1.0 / r if r > 0 else 1.0 for r in od]
        rows = {}
        rows_t = {}
        rows_seen: dict[int, float] = {}
        for tile, waits in waitsets.items():
            after = list(ctx["own"])
            after[tile] -= 1
            rem = list(seen)
            rem[tile] = max(0, rem[tile] - 1)
            unseen = [max(0, CPK - rem[w]) for w in range(TK)]
            tot = sum(unseen)
            ctx2 = dict(ctx)
            ctx2["own"] = after
            ctx2["seen"] = rem
            feat = np.asarray([tile_features(ctx2, w, mode) for w in range(TK)], dtype=np.float64)
            p = predict_fn(feat)
            hat = [max(0.0, min(float(CPK), p[w])) for w in range(TK)]
            v = sum(unseen[w] for w in waits)
            t = sum(max(0, unseen[w] - opp[w]) for w in waits)
            th = sum(max(0.0, unseen[w] - hat[w]) for w in waits)
            prop = 0.0
            for w in waits:
                e = H * (unseen[w] / tot) if tot else 0.0
                prop += max(0.0, unseen[w] - e)
            wsum = sum(unseen[z] * sw[suit_of(z)] for z in range(TK))
            beh = 0.0
            for w in waits:
                share = (unseen[w] * sw[suit_of(w)] / wsum) if wsum else 0.0
                beh += max(0.0, unseen[w] - H * share)
            rows[tile] = (v, t, th, prop, beh)
            if seen_lookup is not None:
                ths = 0.0
                for w in waits:
                    e_s = float(seen_lookup[min(CPK, int(rem[w]))])
                    ths += max(0.0, unseen[w] - e_s)
                rows_seen[tile] = ths
        n += 1
        bt = max(rows, key=lambda k: (rows[k][1], -k))
        bv = max(rows, key=lambda k: (rows[k][0], -k))
        bh = max(rows, key=lambda k: (rows[k][2], -k))
        bp = max(rows, key=lambda k: (rows[k][3], -k))
        bb = max(rows, key=lambda k: (rows[k][4], -k))
        same_hat += bh == bt
        same_v += bv == bt
        same_prop += bp == bt
        same_beh += bb == bt
        if seen_lookup is not None and rows_seen:
            bs = max(rows_seen, key=lambda k: (rows_seen[k], -k))
            same_seen += bs == bt
            if detail is not None:
                detail["seen"].append(bs == bt)
        if detail is not None:
            detail["hat"].append(bh == bt)
            detail["v"].append(bv == bt)
            detail["prop"].append(bp == bt)
            detail["beh"].append(bb == bt)
    _ = rows_t
    return n, same_hat, same_seen, same_v, same_prop, same_beh


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rooms", type=int, default=90)
    ap.add_argument("--test-rooms", type=int, default=60)
    ap.add_argument("--cap-per-room", type=int, default=120)
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--features", choices=["base", "recency"], default="base",
                    help="base=原 34 维聚合；recency=追加弃牌顺序/时间衰减（用户 18:23 假设）")
    args = ap.parse_args(argv)

    tr_rooms = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    rng = random.Random(args.seed)
    rng.shuffle(tr_rooms)
    print(f"提取：train {args.train_rooms} 房 / test {args.test_rooms} 房（seed={args.seed}）", flush=True)
    Xtr, Ytr, _, utr, _ = scan(args.train_rooms, args.cap_per_room, args.seed, args.features)
    # 测试房用不同 seed，避免与训练重叠
    Xte, Yte, ev, ute, tite = scan(args.test_rooms, args.cap_per_room, args.seed + 1, args.features)
    print(f"  特征模式={args.features} 维数={Xtr.shape[1]} · train 房={utr} 行={len(Ytr)} · "
          f"test 房={ute} 行={len(Yte)} · 评估点={len(ev)}", flush=True)
    if len(Ytr) == 0 or len(ev) == 0:
        print("样本不足，退出")
        return 1

    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=args.trees, learning_rate=0.08, max_depth=6, random_state=args.seed
    )
    model.fit(Xtr, Ytr)
    pred = model.predict(Xte)
    mae = float(np.mean(np.abs(pred - Yte)))
    r = float(np.corrcoef(pred, Yte)[0, 1])
    print(f"  [占用回归] test MAE={mae:.3f} 张 · Pearson r={r:.3f}（标签 0..4）", flush=True)

    # ── B' 20:00 要求的泄漏/重叠核对 ──
    # (i) 训练房与测试房文件是否重叠（scan 用了不同 seed，理论上不重叠；此处实证）
    tr_files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    _r = random.Random(args.seed)
    _r.shuffle(tr_files)
    tr_sel = tr_files[: args.train_rooms]
    te_files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    _r = random.Random(args.seed + 1)
    _r.shuffle(te_files)
    te_sel = te_files[: args.test_rooms]
    tr_keys = {Path(p).parent.parent.name for p in tr_sel}
    te_keys = {Path(p).parent.parent.name for p in te_sel}
    tr_names = {Path(p).name for p in tr_sel}
    te_names = {Path(p).name for p in te_sel}
    print(
        f"  [房间重叠核对] 训练房 {len(tr_sel)} 个 / 测试房 {len(te_sel)} 个 · "
        f"按房间目录重合 {len(tr_keys & te_keys)} 个 · 按文件名重合 {len(tr_names & te_names)} 个",
        flush=True,
    )
    n_tr_rows = len(Ytr)
    print(
        f"  [样本重叠核对] train_seen 行数={n_tr_rows} · test 行数={len(Yte)} · "
        f"（scan 用 seed/seed+1 两套独立抽样）",
        flush=True,
    )

    # 可证伪的「只复制 seen」基线：E[opp|seen] 查表（seen 在特征第 1 列，已 /CPK）
    def _lookup_from(X, Y):
        idx = np.clip(np.rint(X[:, 1] * CPK).astype(int), 0, CPK)
        return np.array(
            [float(Y[idx == k].mean()) if (idx == k).any() else 0.0 for k in range(CPK + 1)]
        )

    seen_lookup = _lookup_from(Xtr, Ytr)  # 用训练集构建（原口径，含泄漏）
    seen_lookup_te = _lookup_from(Xte, Yte)  # 用测试集构建（B' 要求的无泄漏口径）
    print(
        "  [seen查表·train建] E[opp|seen] = "
        + ", ".join(f"{k}:{v:.2f}" for k, v in enumerate(seen_lookup)),
        flush=True,
    )
    print(
        "  [seen查表·test建 ] E[opp|seen] = "
        + ", ".join(f"{k}:{v:.2f}" for k, v in enumerate(seen_lookup_te)),
        flush=True,
    )

    # ── A 14:58 要求的第二条门：偏差方向与量级可核对（诊断用）──
    slope = float(np.cov(pred, Yte)[0, 1] / np.var(Yte)) if float(np.var(Yte)) else float("nan")
    print(
        f"  [偏差诊断] mean(pred)={pred.mean():.3f} vs mean(true)={Yte.mean():.3f} 张 · "
        f"std(pred)={pred.std():.3f} vs std(true)={Yte.std():.3f} · "
        f"回归斜率={slope:.3f}（1=无偏；<1=向均值收缩）",
        flush=True,
    )
    print("  [逐牌种偏差] 牌种: pred 均值 / true 均值 / 偏差 张", flush=True)
    worst: list[tuple[float, int, float, float]] = []
    for w in range(TK):
        mask = tite == w
        if int(mask.sum()) < 20:
            continue
        pm = float(pred[mask].mean())
        tm = float(Yte[mask].mean())
        worst.append((abs(pm - tm), w, pm, tm))
    worst.sort(reverse=True)
    for _d, w, pm, tm in worst[:6]:
        print(f"    w={w:2d}: {pm:.3f} / {tm:.3f} / {pm - tm:+.3f}", flush=True)
    mean_bias = float(np.mean(pred - Yte))
    mean_abs_bias = float(np.mean(np.abs(pred - Yte)))
    print(f"  [总体偏差] 平均偏差={mean_bias:+.4f} 张 · 平均绝对偏差={mean_abs_bias:.3f} 张", flush=True)

    n, sh, ss, sv, sp, sb = score(ev, lambda f: model.predict(f), seen_lookup=seen_lookup,
                                  mode=args.features)
    n2, _, ss_te, _, _, _ = score(ev, lambda f: model.predict(f), seen_lookup=seen_lookup_te,
                                  mode=args.features)
    print("=" * 70)
    print(f"评估点（我方听牌、≥2 候选且听口不同）= {n}")
    print(f"  ① 现有口径 V          与全信息 T 同选：{sv}/{n} = {sv / n:.1%}")
    print(f"  ② 比例摊派（无信息）   与全信息 T 同选：{sp}/{n} = {sp / n:.1%}")
    print(f"  ④ 行为加权（手写读牌） 与全信息 T 同选：{sb}/{n} = {sb / n:.1%}")
    print(f"  ★ 学习占用 hat       与全信息 T 同选：{sh}/{n} = {sh / n:.1%}")
    print(f"  ☆ 仅复制 seen 查表(train建,原口径) 与全信息 T 同选：{ss}/{n} = {ss / n:.1%}")
    print(f"  ☆ 仅复制 seen 查表(test建,无泄漏) 与全信息 T 同选：{ss_te}/{n2} = "
          f"{ss_te / n2:.1%}   （n 应同为 {n}）")
    print(f"  判据：hat 需显著高于 ① 的基线 {sv / n:.1%}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
