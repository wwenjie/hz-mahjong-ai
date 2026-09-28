#!/usr/bin/env python3
"""只读检查：监视中的自对弈 job 是否全部结算（供 automations trigger 调用）。

输出**单行 JSON**（便于 code-mode 解析）：
  {"running":N,"pending":N,"settled":M,"all_settled":bool,
   "sig":"...", "alive":{job_id: bool}, "hb":{job_id: int},
   "rows":[["id","status","总分","t","名次","胡次数"], ...]}

对平台零请求、只读 `notes/experiments.json`（+ 本机 `/proc` 只读探活）。
被监视的 job 用 `--keys` 指定（默认：tenpai-wait-6 / ukeire-hand / ukeire-deep）。

**为什么 sig 里含一个心跳位（`hb`）**：上游 trigger 的停滞判据是「同一 sig 持续
>25 分钟」，而一个合法 job 在当前 6 并发下墙钟可达 **~115 分钟**（实测 sibling
1:54–1:56），于是**任何**慢 job 都会必然触发一次假停滞告警（2026-09-28 16:36 即此）。
修法不改 trigger：让 sig 只在「正在跑的 job 真的有 CPU 进展」时才变化——
`hb = floor(该 job 进程 CPU 秒 / 60)`。进程活着且吃 CPU → sig 每 ~1 分钟变一次，
trigger 的 `since` 不断重置，永不误报；进程被回收（`alive=false`）或卡死（CPU 不涨）
→ sig 冻结 → 25 分钟后照常告警。**sig 语义因此是「状态 + 存活进展」，不再是纯状态。**
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS = ROOT / "notes" / "experiments.json"
DEFAULT_KEYS = ("tenpai-wait-6", "ukeire-hand", "ukeire-deep")
_HB_SECONDS = 60  # 心跳桶宽（CPU 秒）


def fmt(x) -> str:
    if not x:
        return "-"
    return f"{x.get('mean'):+.3f}(t{x.get('t'):+.1f})"


def _proc_table():
    """只读 /proc，返回 {pid: (argv, cpu_ticks)}。取不到就跳过（不吞成 0）。"""
    out = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return out
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "cmdline").read_bytes()
            stat = (d / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not raw:
            continue
        argv = [p for p in raw.split(b"\x00") if p]
        argv = [p.decode("utf-8", "replace") for p in argv]
        # stat 的 comm 可能含空格，取最后一个 ')' 之后再切字段
        tail = stat.rpartition(")")[2].split()
        if len(tail) < 13:
            continue
        try:
            cpu = int(tail[11]) + int(tail[12])  # utime + stime
        except ValueError:
            continue
        out[int(d.name)] = (argv, cpu)
    return out


def _argval(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _probe(job, procs, hz):
    """返回 (alive, hb)。hb=-1 表示没找到该 job 的 worker 进程。"""
    want_t = job.get("treatment")
    want_seeds = {str(s) for s in ((job.get("result") or {}).get("seeds") or job.get("seeds") or [])}
    best = None
    for _pid, (argv, cpu) in procs.items():
        if _argval(argv, "--treatment") != want_t:
            continue
        if want_seeds and _argval(argv, "--seed") not in want_seeds:
            continue
        if best is None or cpu > best:
            best = cpu
    if best is None:
        return False, -1
    return True, best // (hz * _HB_SECONDS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    doc = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    rel = [j for j in doc.get("jobs") or [] if any(k in (j.get("id") or "") for k in keys)]
    running = sum(1 for j in rel if j.get("status") == "running")
    pending = sum(1 for j in rel if j.get("status") == "pending")
    settled = [j for j in rel if j.get("status") in ("done", "skipped", "error", "failed")]

    rows = []
    for j in sorted(rel, key=lambda x: x.get("id", "")):
        m = (j.get("result") or {}).get("metrics") or {}
        rows.append([
            j.get("id"), j.get("status"),
            fmt(m.get("总得分")), fmt(m.get("名次分")), fmt(m.get("胡次数")),
        ])

    # 存活探测（只对 running 的 job）：alive + CPU 心跳桶
    procs = _proc_table()
    try:
        import os
        hz = os.sysconf("SC_CLK_TCK") or 100
    except (ValueError, OSError, AttributeError):
        hz = 100
    alive, hb = {}, {}
    for j in rel:
        if j.get("status") != "running":
            continue
        a, h = _probe(j, procs, hz)
        alive[j.get("id")] = a
        hb[j.get("id")] = h

    sig = "|".join(f"{r[0]}:{r[1]}" for r in rows)
    if hb:
        sig += "|hb:" + ",".join(f"{k}={hb[k]}" for k in sorted(hb))
    out = {
        "running": running,
        "pending": pending,
        "settled": len(settled),
        "total": len(rel),
        "all_settled": running == 0 and pending == 0 and len(rel) > 0,
        "sig": sig,
        "alive": alive,
        "hb": hb,
        "rows": rows,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
