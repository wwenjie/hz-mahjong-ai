#!/usr/bin/env python3
"""Stage A′ v2：生产级两级流水线仿真 + 拟合（回 A 02:00 三个口径问题）。

A 02:00 的要求：
1. 拟合对象 = 两级键（主键权重 + 并列层次级键）——只拟合主键结论不成立
2. 目标 = 逐点一致率；切分按房分组；val/test 分开报
3. 必须报「fitted main + 保留生产并列层」的端到端一致率（同类可比）
新 gate：拟合结果必须不劣于生产 v6 同栏数字（81.4 / 87.6 / 72.2）

方法：
- 用落盘字段复刻生产并列层（policy.py:_choose_discard + _break_ties_by_ukeire，v6 knobs）：
  ① fitted_total = main_total + α×(shanten − shanten_ref) 排序取 top
  ② 并列层分组 = 与 top **shanten 相同**的候选（不是 total 并列！——昨晚勘察的关键发现）
  ③ shanten>3 不动（ukeire_max_shanten=3）；shanten==0 用 wait_copies（wait_aware_tenpai）；
    1≤shanten≤3 按 main_total 降序截前 5（presel5；blocks 未落盘，用 main_total 做代理，如实标注），
    再取 ukeire_exact 最大
- 仿真器保真度自检：α=0 时预测 vs is_v5 的一致率（应接近 100%，差距 = 仿真器信息缺口，如实报告）
"""
from __future__ import annotations

import glob
import json
import random
import sys
from collections import Counter

DATA = "agent/out/stage-a-dataset-chunks/cs200-n2000"


def load():
    pts = []
    for f in sorted(glob.glob(f"{DATA}/chunk-*.json")):
        d = json.load(open(f))
        pts.extend(d["points"])
    return pts


def split_by_room(pts, seed=42, ratios=(0.6, 0.2, 0.2)):
    rooms = sorted({p["room"] for p in pts})
    rng = random.Random(seed)
    rng.shuffle(rooms)
    n = len(rooms)
    tr = set(rooms[: int(n * ratios[0])])
    va = set(rooms[int(n * ratios[0]): int(n * (ratios[0] + ratios[1]))])
    te = set(rooms[int(n * (ratios[0] + ratios[1])):])
    return ([p for p in pts if p["room"] in tr],
            [p for p in pts if p["room"] in va],
            [p for p in pts if p["room"] in te])


def predict_production_style(cands, alpha, sec0, sec13):
    """生产级两级流水线仿真。

    alpha: 主键倾斜系数（fitted = main_total + alpha*(shanten - ref_shanten)）
    sec0:  shanten==0 时的次级键权重 (w_wait_copies, w_ukeire, w_wait_kinds)
    sec13: 1<=shanten<=3 时的次级键权重 (w_wait_copies, w_ukeire, w_wait_kinds)
           生产现状 = sec13 只取 ukeire_exact 最大 ⇒ (0,1,0)；sec0 只取 wait_copies ⇒ (1,0,0)
    """
    ref_sh = max(cands, key=lambda c: c["main_total"])["shanten"]
    scored = [(c["main_total"] + alpha * (c["shanten"] - ref_sh), c) for c in cands]
    scored.sort(key=lambda x: -x[0])
    top = scored[0][1]
    top_sh = top["shanten"]
    group = [c for _, c in scored if c["shanten"] == top_sh]
    if len(group) == 1 or top_sh > 3:
        return top["tile"]
    if top_sh == 0:
        w = sec0
    else:
        w = sec13
        group = group[:5]  # presel5（blocks 未落盘，main_total 降序代理截断——如实标注）
    def key(c):
        wc = c.get("wait_copies")
        ue = c.get("ukeire_exact")
        wk = c.get("wait_kinds")
        return (w[0] * (wc if wc is not None else -1.0)
                + w[1] * (ue if ue is not None else -1.0)
                + w[2] * (wk if wk is not None else -1.0))
    return max(group, key=key)["tile"]


def evaluate(pts, alpha, sec0, sec13):
    """返回 (总, 非并列, 并列, 仿真器保真度)——并列口径 = main_total 顶层并列（与 C 的报告同）。"""
    n = hit = nt_n = nt_hit = t_n = t_hit = 0
    fid_n = fid_hit = 0
    for p in pts:
        cands = p.get("candidates") or []
        if not cands:
            continue
        bot = p.get("bot_tile")
        v5t = p.get("v5_tile")
        if bot is None:
            continue
        pred = predict_production_style(cands, alpha, sec0, sec13)
        n += 1
        hit += (pred == bot)
        fid_n += 1
        fid_hit += (pred == v5t)
        mt = [c["main_total"] for c in cands]
        mx = max(mt)
        is_tie = sum(1 for v in mt if abs(v - mx) < 1e-9) > 1
        if is_tie:
            t_n += 1
            t_hit += (pred == bot)
        else:
            nt_n += 1
            nt_hit += (pred == bot)
    return (hit / n if n else 0, nt_hit / nt_n if nt_n else 0,
            t_hit / t_n if t_n else 0, fid_hit / fid_n if fid_n else 0)


def main():
    pts = load()
    train, val, test = split_by_room(pts)
    print(f"points={len(pts)} train={len(train)} val={len(val)} test={len(test)}")

    PROD_SEC0 = (1.0, 0.0, 0.0)   # wait_aware_tenpai: 听牌看 wait_copies
    PROD_SEC13 = (0.0, 1.0, 0.0)  # 1-3 向听: 精确进张最大

    print("\n== 仿真器保真度自检（α=0, 生产次级键）==")
    a, nt, t, fid = evaluate(val, 0.0, PROD_SEC0, PROD_SEC13)
    print(f"val: 与 bot 一致率 {a:.4f}（非并列 {nt:.4f} / 并列 {t:.4f}）")
    print(f"val: 仿真器 vs 生产 is_v5 保真度 {fid:.4f}（缺口=blocks 未落盘+其余决策逻辑）")
    a, nt, t, fid = evaluate(test, 0.0, PROD_SEC0, PROD_SEC13)
    print(f"test: 与 bot 一致率 {a:.4f}（非并列 {nt:.4f} / 并列 {t:.4f}）；保真度 {fid:.4f}")

    print("\n== 拟合（val 上坐标下降）==")
    best = None
    # α 网格
    alphas = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    # 次级键网格（对两个 shanten 层分别）
    sec_grids = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.25, 0.0), (1.0, 0.25, 0.25),
                 (0.0, 1.0, 0.25), (0.5, 1.0, 0.0), (1.0, 0.5, 0.25), (0.25, 1.0, 0.25)]
    cur = (0.0, PROD_SEC0, PROD_SEC13)
    best_val = evaluate(val, *cur)[0]
    print(f"[init] α=0 生产次级键: val {best_val:.4f}")
    for rnd in range(8):
        improved = False
        for av in alphas:
            if av == cur[0]:
                continue
            cand = (av, cur[1], cur[2])
            v = evaluate(val, *cand)[0]
            if v > best_val:
                best_val, cur, improved = v, cand, True
                print(f"[r{rnd}] α→{av}: val {v:.4f}")
        for layer in (1, 2):
            for gv in sec_grids:
                if gv == cur[layer]:
                    continue
                cand = list(cur)
                cand[layer] = gv
                cand = tuple(cand)
                v = evaluate(val, *cand)[0]
                if v > best_val:
                    best_val, cur, improved = v, cand, True
                    print(f"[r{rnd}] sec{layer}→{gv}: val {v:.4f}")
        if not improved:
            print(f"[r{rnd}] 无改进，收敛")
            break

    alpha, sec0, sec13 = cur
    print(f"\n拟合结果: α={alpha} sec0={sec0} sec13={sec13}")
    va, vnt, vt, vfid = evaluate(val, alpha, sec0, sec13)
    ta, tnt, tt, tfid = evaluate(test, alpha, sec0, sec13)
    print("\n" + "=" * 64)
    print("Stage A′ v2（生产级两级流水线；A 02:00 新 gate：不劣于 v6 同栏）")
    print("=" * 64)
    print(f"{'':<14}{'总体':>10}{'非并列':>10}{'并列':>10}")
    print(f"{'生产 v6':<14}{'81.4%':>10}{'87.6%':>10}{'72.2%':>10}  (C 00:20)")
    print(f"{'仿真器 α=0':<14}{ta and evaluate(test,0.0,PROD_SEC0,PROD_SEC13)[0] or 0:>10.4f}"
          f"{evaluate(test,0.0,PROD_SEC0,PROD_SEC13)[1]:>10.4f}"
          f"{evaluate(test,0.0,PROD_SEC0,PROD_SEC13)[2]:>10.4f}")
    print(f"{'拟合 val':<14}{va:>10.4f}{vnt:>10.4f}{vt:>10.4f}")
    print(f"{'拟合 test':<14}{ta:>10.4f}{tnt:>10.4f}{tt:>10.4f}")
    print(f"\n判据 A: test 总体 ≥84%: {ta:.4f} {'≥ ⇒ Stage B′' if ta >= 0.84 else '< ⇒ 不过'}")
    gate = (ta >= 0.814 and tnt >= 0.876 and tt >= 0.722)
    print(f"判据 B（A 新 gate，不劣于 v6 同栏）: {'过' if gate else '不过'}"
          f"（总 {ta:.4f} vs .814 / 非并列 {tnt:.4f} vs .876 / 并列 {tt:.4f} vs .722）")


if __name__ == "__main__":
    sys.exit(main())
