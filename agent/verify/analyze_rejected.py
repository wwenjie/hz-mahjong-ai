#!/usr/bin/env python3
"""只读：真机 `action.rejected` 成因分析（对外零请求）。

口径修正（踩过的坑）：`logs/*.jsonl` 里混着**本地试跑**（`run1/run2/run3/run-firstlegal.jsonl`）
与**测试房**（game_id 前缀 `t_`，如 `t_0cfde5a00075`）。不剔除就会把 835 次拒绝当成真机问题——
实际**真机只有 503 次**，另 333 次是测试房。**按房分层，不按文件。**

回答三件事：
  1. 真机被拒动作分布（按 phase / 动作族）
  2. 被拒后是否紧跟同族 submitted（= 重试自愈）
  3. 被拒 `hu` 的那一局最终结局（是否仍由我们胡 → 判断是否功能性损失）

用法: nice -n 19 uv run python agent/verify/analyze_rejected.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
from pathlib import Path

REAL_FILE = re.compile(r"a_[0-9a-f]{12}\.jsonl$")
OUR_UID = "u_a7f7c67bb14a"
ROOT = Path(__file__).resolve().parent.parent.parent


def parse(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            i = line.find("{")
            if i < 0:
                continue
            try:
                yield json.loads(line[i:])
            except json.JSONDecodeError:
                continue


def family(a: str) -> str:
    a = (a or "").strip()
    for fam in ("discard", "chi", "peng", "gang", "hu", "pass", "draw"):
        if a.startswith(fam):
            return fam
    return a.split(":", 1)[0] or "?"


def is_real(path: str) -> bool:
    return bool(REAL_FILE.search(os.path.basename(path)))


def round_outcomes(room: str, gid: str):
    """从事件流副本读该局每小局的胡家座位，返回 {round_no: (seat, detail)}。"""
    side = ROOT / "data" / "auto_sessions" / room / "events" / f"{gid}.json"
    if not side.exists():
        return None
    doc = json.loads(side.read_text(encoding="utf-8"))
    seats = doc.get("seats") or []
    our = next((i for i, s in enumerate(seats) if s.get("user_id") == OUR_UID), None)
    ends = {}
    for b in doc.get("blocks", []):
        for e in (b.get("events") or []):
            if e.get("type") == "round_ended":
                d = e.get("data") or {}
                ends[d.get("round_no")] = (e.get("seat"), d.get("detail"), d.get("draw"))
    return our, ends


def main() -> int:
    all_files = sorted(glob.glob("logs/*.jsonl"))
    real_files = [p for p in all_files if is_real(p)]
    print(f"日志文件 {len(all_files)}，其中真机 {len(real_files)}，"
          f"测试房/试跑 {len(all_files) - len(real_files)}\n")

    rej_phase = collections.Counter()
    rej_fam = collections.Counter()
    rej_total = sub_total = 0
    healed = unhealed = 0
    hu_rej = []

    for path in real_files:
        seq = list(parse(path))
        for e in seq:
            ev = e.get("event")
            if ev == "action.rejected":
                rej_total += 1
                rej_phase[str(e.get("phase"))] += 1
                rej_fam[family(str(e.get("action")))] += 1
                if family(str(e.get("action"))) == "hu":
                    hu_rej.append((path, e.get("game_id"), e.get("phase"), e.get("round_no")))
            elif ev == "action.submitted":
                sub_total += 1
        by_game = collections.defaultdict(list)
        for e in seq:
            if e.get("event") in ("action.rejected", "action.submitted"):
                by_game[e.get("game_id")].append(e)
        for gid, evs in by_game.items():
            for idx, e in enumerate(evs):
                if e.get("event") != "action.rejected":
                    continue
                fam = family(str(e.get("action")))
                nxt = next((x for x in evs[idx + 1:] if family(str(x.get("action"))) == fam), None)
                if nxt is not None and nxt.get("event") == "action.submitted":
                    healed += 1
                else:
                    unhealed += 1

    print("=== 1) 真机被拒分布 ===")
    print(f"rejected {rej_total} / submitted {sub_total} → 拒绝率 {rej_total / max(sub_total,1):.4%}")
    print("按 phase:", dict(rej_phase.most_common()))
    print("按动作族:", dict(rej_fam.most_common()))
    print()
    print("=== 2) 自愈（拒绝后是否紧跟同族 submitted）===")
    tot = healed + unhealed
    print(f"自愈 {healed}/{tot} = {healed / max(tot,1):.2%}；未自愈 {unhealed}")
    print()
    print("=== 3) hu 被拒的后果（该小局最终是否由我们胡）===")
    print(f"hu 被拒 {len(hu_rej)} 次")
    loss = 0
    for path, gid, ph, rn in hu_rej:
        room = os.path.basename(path).replace(".jsonl", "")
        out = round_outcomes(room, gid)
        if out is None:
            print(f"  {os.path.basename(path):24s} {gid} phase={ph}（无事件流副本）")
            continue
        our, ends = out
        rn_key = None
        for k in ends:
            if str(k) == str(rn):
                rn_key = k
        seat, detail, draw = ends.get(rn_key, (None, None, None))
        if seat == our:
            verdict = "我们仍胡了（无损失）"
        elif draw:
            verdict = "该局流局"
            loss += 1
        else:
            verdict = f"被别人胡(座位{seat})— 疑似丢胡"
            loss += 1
        print(f"  {os.path.basename(path):24s} {gid} 被拒于第{rn}小局 我们座位={our} "
              f"→ {verdict} detail={detail}")
    print(f"\n结论：疑似丢胡/流局 {loss}/{len(hu_rej)}；其余均仍由我们胡。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
