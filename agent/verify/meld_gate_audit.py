#!/usr/bin/env python
"""副露率诊断（口径 B 修正版，零 src 改动）：**直接用线上 HeuristicDecider 审计 legacy 闸门**。

承接 A 00:10 裁决 + 我 00:20 更正（冠军档 v5/v6 route_aware=False，走 legacy 分支）。

**方法（保真度最高）**：不重写判据，直接实例化线上 `HeuristicDecider`（`PolicyConfig` 默认档
= v6 冠军档的副露判据：meld_tolerance=STRICT、pair_route_pairs=5），在真机事件流的每个
响应窗上：
1. `replay.situation_for` 构造 Situation（PENG 窗 / CHI 窗分开，CHI 仅上家）；
2. `rules.action.legal_actions` 算合法动作集；
3. 问 `decider._choose_response`（**线上原样代码**）会选什么；
4. 同时问一个 `meld_tolerance=EQUAL` 的反事实 decider；
5. 与实际动作（下一事件）对照。

**回答 A 的三条可能性**：
- 闸门问题 ⇒ 「STRICT 拒、EQUAL 接」的窗口多（放开闸门副露率就能升）；
- 估值问题 ⇒ 七对守门（_is_pair_route）拦截的窗口多；
- 结构性问题 ⇒ 合法窗口本来就少（无对子/无搭子）。

用法：`.venv/bin/python agent/verify/meld_gate_audit.py [--rooms N] --arm-map agent/out/arm_map.json --arm v5`
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_RESPONSE_CHI, PHASE_RESPONSE_PENG  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, MeldTolerance, PolicyConfig  # noqa: E402

OUR = "u_a7f7c67bb14a"
MELD_KINDS = {"chi", "peng", "gang", "minggang", "angang"}


def _tile_of(raw: object) -> int | None:
    if raw in (None, ""):
        return None
    try:
        return tiles.parse(str(raw))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--arm-map", default=None)
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()

    arm_of = {}
    if args.arm_map:
        with open(args.arm_map, encoding="utf-8") as f:
            arm_of = json.load(f).get("map", {})

    files = sorted(glob.glob(str(REPO / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if arm_of:
        keep = []
        for p in files:
            gid = pathlib.Path(p).stem
            arm = arm_of.get(gid)
            if arm is None or arm == "mixed":
                continue
            if args.arm and arm != args.arm:
                continue
            keep.append(p)
        files = keep
    if args.rooms:
        files = files[:: max(1, len(files) // args.rooms)][: args.rooms]

    # 两个 decider：STRICT（=线上冠军档判据）与 EQUAL（反事实）
    dec_strict = HeuristicDecider(PolicyConfig())
    dec_equal = HeuristicDecider(PolicyConfig(meld_tolerance=MeldTolerance.EQUAL))

    rooms = 0
    skipped = 0
    # 每个窗口记录一行统计
    st = {
        "peng_windows": 0, "peng_legal": 0,
        "chi_windows": 0, "chi_legal": 0,   # chi 窗仅上家
        "actual_meld": 0,                   # 实际副露（我方下一事件是 chi/peng/gang）
        # 在「legal 含副露动作」的窗口上：
        "legal_meld_windows": 0,
        "strict_takes": 0,                  # STRICT decider 选择副露
        "equal_takes": 0,                   # EQUAL decider 选择副露
        "strict_reject_reasons": collections.Counter(),  # STRICT 拒时的 last_reason 前缀
        "pair_gate_blocks": 0,              # 七对守门拦截（reason 含「七对」）
        "equal_extra": 0,                   # EQUAL 接而 STRICT 拒（=放开闸门的潜在增量）
        "equal_extra_actual": 0,            # 其中真机上实际也没副露（纯被闸门卡掉）
    }

    for path in files:
        try:
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        me = ids.index(OUR)
        rooms += 1
        try:
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            for i, ev in enumerate(events):
                # 先 apply 保持 state 推进
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                if ev.get("type") != replay.DISCARDED:
                    continue
                seat = ev.get("seat")
                if seat == me or not isinstance(seat, int):
                    continue
                offered = _tile_of(ev.get("tile"))
                if offered is None or tiles.is_god(offered):
                    continue
                # 我方实际动作：下一事件是我方 chi/peng/gang ⇒ 实际副露
                nxt = events[i + 1] if i + 1 < len(events) else None
                actual = bool(
                    nxt and nxt.get("seat") == me and str(nxt.get("type", "")).lower() in MELD_KINDS
                )
                # 若下一事件是 hu / round_end（有人胡了），窗口实际未对我结算——跳过
                if nxt and str(nxt.get("type", "")).lower() in ("hu", "round_end", "win"):
                    continue

                # 两个窗口：PENG（任何对手打出）+ CHI（仅上家=(me-1)%4 打出）
                phases = [(PHASE_RESPONSE_PENG, "peng")]
                if seat == (me - 1) % 4:
                    phases.append((PHASE_RESPONSE_CHI, "chi"))
                for phase, wname in phases:
                    try:
                        sit = state.situation_for(me, phase=phase, offered=offered, responding=(me,))
                    except Exception:  # noqa: BLE001
                        skipped += 1
                        continue
                    st[f"{wname}_windows"] += 1
                    try:
                        acts = legal_actions(sit)
                    except Exception:  # noqa: BLE001
                        skipped += 1
                        continue
                    has_meld = any(a.kind in ("peng", "chi") for a in acts)
                    if not has_meld:
                        continue
                    st[f"{wname}_legal"] += 1
                    st["legal_meld_windows"] += 1
                    if actual:
                        st["actual_meld"] += 1
                    # STRICT 判
                    try:
                        a_strict = dec_strict._choose_response(sit, acts)
                    except Exception:  # noqa: BLE001
                        skipped += 1
                        continue
                    takes_strict = a_strict.kind in ("peng", "chi")
                    if takes_strict:
                        st["strict_takes"] += 1
                    else:
                        reason = getattr(dec_strict, "last_reason", "") or ""
                        st["strict_reject_reasons"][reason.split("：")[0][:20]] += 1
                        if "七对" in reason:
                            st["pair_gate_blocks"] += 1
                    # EQUAL 反事实
                    try:
                        a_equal = dec_equal._choose_response(sit, acts)
                    except Exception:  # noqa: BLE001
                        continue
                    takes_equal = a_equal.kind in ("peng", "chi")
                    if takes_equal:
                        st["equal_takes"] += 1
                    if takes_equal and not takes_strict:
                        st["equal_extra"] += 1
                        if not actual:
                            st["equal_extra_actual"] += 1

    print(f"房={rooms} 跳过={skipped}")
    print(f"\n响应窗：PENG {st['peng_windows']}（含合法碰 {st['peng_legal']}）· CHI（仅上家）{st['chi_windows']}（含合法吃 {st['chi_legal']}）")
    lw = st["legal_meld_windows"]
    print(f"\n含合法副露动作的窗口：{lw}")
    if lw:
        print(f"  实际副露（真机）：{st['actual_meld']} ({st['actual_meld']/lw:.1%})")
        print(f"  STRICT（线上判据）会副露：{st['strict_takes']} ({st['strict_takes']/lw:.1%})")
        print(f"  EQUAL（放开档）会副露：{st['equal_takes']} ({st['equal_takes']/lw:.1%})")
        print(f"  ★ EQUAL 接而 STRICT 拒：{st['equal_extra']} ({st['equal_extra']/lw:.1%}) —— 放开闸门的潜在增量")
        print(f"    其中真机实际也没副露：{st['equal_extra_actual']}（纯被闸门卡掉的）")
        print(f"  七对守门拦截：{st['pair_gate_blocks']} ({st['pair_gate_blocks']/lw:.1%})")
        print(f"\n  STRICT 拒的理由分布：")
        for r, c in st["strict_reject_reasons"].most_common(8):
            print(f"    {c:5d}  {r}")
        print(f"\n  ★ 判读：")
        print(f"    - equal_extra/legal 占比高 ⇒ 闸门问题（放开 EQUAL 副露率即升）")
        print(f"    - pair_gate_blocks/legal 占比高 ⇒ 估值问题（七对守门拦的）")
        print(f"    - strict_takes ≈ actual ⇒ 探针与线上口径一致（自校验）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
