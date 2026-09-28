#!/usr/bin/env python3
"""只读检查：监视中的自对弈 job 是否全部结算（供 automations trigger 调用）。

输出**单行 JSON**（便于 code-mode 解析）：
  {"running":N,"pending":N,"settled":M,"all_settled":bool,
   "sig":"...", "rows":[["id","status","总分","t","名次","胡次数"], ...]}

对平台零请求、只读 `notes/experiments.json`。被监视的 job 用 `--keys` 指定
（默认：tenpai-wait-6 / ukeire-hand / ukeire-deep）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS = ROOT / "notes" / "experiments.json"
DEFAULT_KEYS = ("tenpai-wait-6", "ukeire-hand", "ukeire-deep")


def fmt(x) -> str:
    if not x:
        return "-"
    return f"{x.get('mean'):+.3f}(t{x.get('t'):+.1f})"


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
    sig = "|".join(f"{r[0]}:{r[1]}" for r in rows)
    out = {
        "running": running,
        "pending": pending,
        "settled": len(settled),
        "total": len(rel),
        "all_settled": running == 0 and pending == 0 and len(rel) > 0,
        "sig": sig,
        "rows": rows,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
