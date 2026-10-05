#!/usr/bin/env python3
"""Stage A′：数据驱动拟合主导项权重（A 22:15 立项，两级键 23:05 扩范围）。

目标：用「与强 bot 一致率」为目标函数，拟合出牌排序键的少数系数。

数据底座：C 的 `stage_a_dataset_export.py` 产物（chunk-*.json）。
Schema（`stage_a_decision_point.v1`）：
  point 级：room / round_no / features[29] / god_n / shanten / bot_tile / v5_tile / agree / candidates[]
  候选级：tile / main_total / shanten / is_bot / is_v5 / wait_copies / ukeire_exact / wait_kinds

关键口径（C 22:45）：子项（route_value/feed/god_penalty）**不落盘**——由 total 合成。
⇒ 主键层拟合只能做「倾斜」形式：fitted_total = main_total + alpha×(shanten − shanten_at_main_total_argmax)。
  alpha=0 ⇒ 逐位等于 v5 主键；alpha>0 ⇒ 偏好低向听；alpha<0 ⇒ 偏好高向听。
  这是「改主导项权重」的有限实现（铁律第①条），在现有数据下唯一可行。
次级键（并列层）：s1×wait_copies + s2×ukeire_exact + s3×wait_kinds（null 按 0 处理，只在该点至少一个候选非空时启用）。

切分纪律（A 22:15②）：按房分组切分 train/val/test（默认 60/20/20），seed 可复现。
对照（A 22:15②）：val 与 test 同时报；差大 = 过拟合。
判据（A 22:15②）：test 一致率 ≥84%（基线 81.4%）⇒ 进 Stage B′；否则直上全量 SL。
报告拆两栏（A 23:05②）：非并列点 vs 并列点分别报一致率。

用法：
    uv run python tools/stage_a_prime_fit.py \
        --data agent/out/stage-a-dataset-chunks/cs200-n2000 \
        [--out agent/out/stage-a-prime.txt] [--seed 42] [--max-rounds 20]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from dataclasses import dataclass


# ---------- 数据结构 ----------

@dataclass
class Candidate:
    tile: int
    main_total: float
    shanten: float
    is_bot: bool
    is_v5: bool
    wait_copies: float | None
    ukeire_exact: float | None
    wait_kinds: float | None


@dataclass
class DecisionPoint:
    room_id: str
    round_no: int
    candidates: list[Candidate]
    features: list[float] | None = None


# ---------- 加载 ----------

def load_dataset(path: str) -> list[DecisionPoint]:
    """加载 C 的数据底座（chunk-*.json 目录或单文件）。"""
    files = sorted(glob.glob(f"{path}/chunk-*.json")) if not path.endswith(".json") else [path]
    if not files:
        raise FileNotFoundError(f"no chunk-*.json under {path}")
    points: list[DecisionPoint] = []
    for fp in files:
        d = json.load(open(fp))
        for p in d.get("points", []):
            cands = [
                Candidate(
                    tile=c["tile"],
                    main_total=c["main_total"],
                    shanten=c["shanten"],
                    is_bot=c["is_bot"],
                    is_v5=c["is_v5"],
                    wait_copies=c.get("wait_copies"),
                    ukeire_exact=c.get("ukeire_exact"),
                    wait_kinds=c.get("wait_kinds"),
                )
                for c in p.get("candidates", [])
            ]
            if not cands:
                continue
            points.append(DecisionPoint(
                room_id=p["room"],
                round_no=p["round_no"],
                candidates=cands,
                features=p.get("features"),
            ))
    return points


# ---------- 切分 ----------

def split_by_room(points: list[DecisionPoint], seed: int = 42,
                  ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
                  ) -> tuple[list[DecisionPoint], list[DecisionPoint], list[DecisionPoint]]:
    rooms = sorted({p.room_id for p in points})
    rng = random.Random(seed)
    rng.shuffle(rooms)
    n = len(rooms)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_rooms = set(rooms[:n_train])
    val_rooms = set(rooms[n_train:n_train + n_val])
    test_rooms = set(rooms[n_train + n_val:])
    return (
        [p for p in points if p.room_id in train_rooms],
        [p for p in points if p.room_id in val_rooms],
        [p for p in points if p.room_id in test_rooms],
    )


# ---------- 两级键预测 ----------

TIE_EPS = 1e-9


def _fitted_main(c: Candidate, alpha: float, ref_shanten: float) -> float:
    """主键倾斜：main_total + alpha×(shanten − ref_shanten)。alpha=0 时逐位等于 v5。"""
    return c.main_total + alpha * (c.shanten - ref_shanten)


def _tie_key(c: Candidate, s: tuple[float, float, float]) -> float:
    """次级键：s1×wait_copies + s2×ukeire_exact + s3×wait_kinds（null→0）。"""
    wc = c.wait_copies if c.wait_copies is not None else 0.0
    ue = c.ukeire_exact if c.ukeire_exact is not None else 0.0
    wk = c.wait_kinds if c.wait_kinds is not None else 0.0
    return s[0] * wc + s[1] * ue + s[2] * wk


def predict(point: DecisionPoint, alpha: float, s: tuple[float, float, float]) -> int:
    """两级键预测：先主键（倾斜后）取 max，并列（|差|<eps）进入次级键再取 max。"""
    ref_shanten = max(point.candidates, key=lambda c: c.main_total).shanten
    scored = [(_fitted_main(c, alpha, ref_shanten), c) for c in point.candidates]
    best_main = max(sc for sc, _ in scored)
    tied = [c for sc, c in scored if abs(sc - best_main) < TIE_EPS]
    if len(tied) == 1:
        return tied[0].tile
    # 次级键只在至少一个候选非空时启用；全空则按 main_total 顺序（稳定）
    if any(c.wait_copies is not None or c.ukeire_exact is not None or c.wait_kinds is not None for c in tied):
        return max(tied, key=lambda c: _tie_key(c, s)).tile
    return tied[0].tile


# ---------- 评估 ----------

def agreement(points: list[DecisionPoint], alpha: float, s: tuple[float, float, float],
              ) -> tuple[float, float, float]:
    """返回 (总一致率, 非并列点一致率, 并列点一致率)。并列判定用 v5 主键（alpha=0）。"""
    n = agree = 0
    n_tie = agree_tie = 0
    n_nt = agree_nt = 0
    for p in points:
        bot_tile = next((c.tile for c in p.candidates if c.is_bot), None)
        if bot_tile is None:
            continue
        pred = predict(p, alpha, s)
        hit = (pred == bot_tile)
        # 并列判定：v5 主键（alpha=0）下是否并列
        ref_shanten = max(p.candidates, key=lambda c: c.main_total).shanten
        scored = [(_fitted_main(c, 0.0, ref_shanten), c) for c in p.candidates]
        best_main = max(sc for sc, _ in scored)
        tied = [c for sc, c in scored if abs(sc - best_main) < TIE_EPS]
        is_tie = len(tied) > 1
        n += 1
        agree += hit
        if is_tie:
            n_tie += 1
            agree_tie += hit
        else:
            n_nt += 1
            agree_nt += hit
    return (
        agree / n if n else 0.0,
        agree_nt / n_nt if n_nt else 0.0,
        agree_tie / n_tie if n_tie else 0.0,
    )


# ---------- 拟合 ----------

def coordinate_descent(val: list[DecisionPoint], max_rounds: int = 20, verbose: bool = True,
                       ) -> tuple[float, tuple[float, float, float], float]:
    """坐标下降：主键 alpha + 次级键 (s1, s2, s3)。目标函数 = val 一致率。"""
    alpha = 0.0
    s = (1.0, 0.0, 0.0)  # 默认：wait_copies 主导（v5 现状）
    grids = {
        "alpha": [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
        "s1": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        "s2": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        "s3": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    }
    best_val, _, _ = agreement(val, alpha, s)
    if verbose:
        base_val, base_nt, base_t = agreement(val, 0.0, (1.0, 0.0, 0.0))
        print(f"[init] v5 基线 val 一致率 {base_val:.4f}（非并列 {base_nt:.4f} / 并列 {base_t:.4f}）")
    for rnd in range(max_rounds):
        improved = False
        # 主键 alpha
        for v in grids["alpha"]:
            if v == alpha:
                continue
            val_acc, _, _ = agreement(val, v, s)
            if val_acc > best_val:
                best_val = val_acc
                alpha = v
                improved = True
                if verbose:
                    print(f"[round {rnd}] alpha → {v}: val {val_acc:.4f}")
        # 次级键 s1/s2/s3
        for idx, key in enumerate(("s1", "s2", "s3")):
            for v in grids[key]:
                if v == s[idx]:
                    continue
                s_try = list(s)
                s_try[idx] = v
                s_try = tuple(s_try)
                val_acc, _, _ = agreement(val, alpha, s_try)
                if val_acc > best_val:
                    best_val = val_acc
                    s = s_try
                    improved = True
                    if verbose:
                        print(f"[round {rnd}] {key} → {v}: val {val_acc:.4f}")
        if not improved:
            if verbose:
                print(f"[round {rnd}] 无改进，收敛")
            break
    return alpha, s, best_val


# ---------- 主入口 ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="C 的数据底座路径（chunk-*.json 目录）")
    ap.add_argument("--out", default="agent/out/stage-a-prime.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=20)
    args = ap.parse_args()

    points = load_dataset(args.data)
    print(f"加载决策点 {len(points)}（按房分组切分纪律：A 22:15②）")
    train, val, test = split_by_room(points, seed=args.seed)
    print(f"train {len(train)} / val {len(val)} / test {len(test)}")

    # 基线（alpha=0, s=(1,0,0)）
    base_val, base_nt, base_t = agreement(val, 0.0, (1.0, 0.0, 0.0))
    base_test, base_test_nt, base_test_t = agreement(test, 0.0, (1.0, 0.0, 0.0))
    print(f"v5 基线   val : {base_val:.4f}（非并列 {base_nt:.4f} / 并列 {base_t:.4f}）")
    print(f"v5 基线   test: {base_test:.4f}（非并列 {base_test_nt:.4f} / 并列 {base_test_t:.4f}）")

    # 拟合
    alpha, s, best_val = coordinate_descent(val, max_rounds=args.max_rounds)
    print(f"\n拟合权重：alpha={alpha} / 次级键 s={s}")

    # 对照（A 22:15②）：val 与 test 同时报；拆两栏（A 23:05②）
    val_acc, val_nt, val_t = agreement(val, alpha, s)
    test_acc, test_nt, test_t = agreement(test, alpha, s)
    lines = [
        "=" * 60,
        "Stage A′ 结果（预登记判据：test 一致率 ≥84% ⇒ 进 Stage B′）",
        "=" * 60,
        f"v5 基线   val : {base_val:.4f}（非并列 {base_nt:.4f} / 并列 {base_t:.4f}）",
        f"v5 基线   test: {base_test:.4f}（非并列 {base_test_nt:.4f} / 并列 {base_test_t:.4f}）",
        f"拟合权重  val : {val_acc:.4f}（非并列 {val_nt:.4f} / 并列 {val_t:.4f}）",
        f"拟合权重  test: {test_acc:.4f}（非并列 {test_nt:.4f} / 并列 {test_t:.4f}）",
        f"val-test 差: {val_acc - test_acc:+.4f}（差大 = 过拟合信号）",
        f"判据: test {test_acc:.4f} {'≥' if test_acc >= 0.84 else '<'} 0.84 ⇒ "
        f"{'进 Stage B′（接成臂 A/B）' if test_acc >= 0.84 else '判线性键空间不足，直上全量 SL'}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
        fh.write(f"\nalpha: {alpha}\n")
        fh.write(f"次级键权重 (s1,s2,s3): {s}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
