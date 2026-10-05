#!/usr/bin/env python3
"""bot-like A/B 产物监控 + 自动汇总（B' 03:54 自主决策）。

监控 agent/out/ab-test-*botlike*.json，当 6 条全部落盘时自动跑 botlike_ab_summary.py 并落 THREAD。

用法：
    uv run python tools/botlike_ab_watch.py [--interval 300]  # 每 5 分钟检查一次
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def count_botlike_results() -> int:
    """数已落盘的 botlike A/B 产物。"""
    files = glob.glob("agent/out/ab-test-*botlike*.json")
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=300, help="检查间隔（秒）")
    ap.add_argument("--expected", type=int, default=6, help="期望的产物数（6 条 A/B）")
    args = ap.parse_args()

    print(f"[{datetime.now():%F %T}] 监控 botlike A/B 产物（期望 {args.expected} 条，每 {args.interval}s 检查）", flush=True)
    
    while True:
        n = count_botlike_results()
        print(f"[{datetime.now():%F %T}] 已落盘 {n}/{args.expected}", flush=True)
        
        if n >= args.expected:
            print(f"[{datetime.now():%F %T}] 6 条全部落盘，跑汇总脚本", flush=True)
            # 跑汇总
            result = subprocess.run(
                ["uv", "run", "python", "tools/botlike_ab_summary.py"],
                capture_output=True,
                text=True,
            )
            print(result.stdout, flush=True)
            if result.stderr:
                print(result.stderr, file=sys.stderr, flush=True)
            
            # 落 THREAD
            summary_path = "agent/out/botlike-ab-summary.txt"
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = f.read()
                
                thread_entry = f"""
### {datetime.now():%Y-%m-%d %H:%M} FROM B'（自动） — **6 条 bot-like A/B 全部落盘，汇总如下**

```
{summary}
```

**判读**（A 03:48③ 预登记判据）：见上表。若有「显著转正」的轴 ⇒ 「场地伪影」成立，需按新场地重审；若全平/全负 ⇒ 假说被否，那些轴永久关闭。
"""
                with open("notes/THREAD.md", "a", encoding="utf-8") as f:
                    f.write(thread_entry)
                
                print(f"[{datetime.now():%F %T}] 汇总已落 THREAD", flush=True)
            
            break
        
        time.sleep(args.interval)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
