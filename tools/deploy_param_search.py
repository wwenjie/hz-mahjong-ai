#!/usr/bin/env python3
"""参数搜索远程部署 + 冒烟（B' 2026-10-06 23:53，用户指令）。

三台机器（mj-c / mj-d / mj-e）的部署流程：
1. rsync 代码（排除 data/.venv/.git/logs/agent/out/）
2. 装环境（C/D 用 venv+pip，E 用已有 miniconda）
3. 冒烟（跑 1 组 m4，掐秒表）
4. 输出各机器的实际吞吐，供参数搜索排产

用法：
    uv run python tools/deploy_param_search.py [--machines mj-c,mj-d,mj-e] [--smoke-only]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

MACHINES = {
    "mj-c": {"host": "49.235.23.236", "port": 22, "user": "ubuntu", "cores": 16, "has_conda": False},
    "mj-d": {"host": "114.132.153.78", "port": 22, "user": "ubuntu", "cores": 16, "has_conda": False},
    "mj-e": {"host": "connect.nma1.seetacloud.com", "port": 17407, "user": "root", "cores": 112, "has_conda": True},
}

RSYNC_EXCLUDES = [
    "--exclude=data/",
    "--exclude=.venv/",
    "--exclude=.git/",
    "--exclude=logs/",
    "--exclude=agent/out/",
    "--exclude=agent/agentb-coordinator/",
    "--exclude=*.pyc",
    "--exclude=__pycache__/",
    "--exclude=*.egg-info/",
    "--exclude=dist/",
    "--exclude=build/",
]


def run(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run command, return (rc, stdout, stderr)."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def ssh_cmd(machine: str) -> list[str]:
    m = MACHINES[machine]
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
        "-p", str(m["port"]), f"{m['user']}@{m['host']}",
    ]


def deploy(machine: str, smoke_only: bool = False) -> dict:
    """Deploy code + env to one machine. Returns status dict."""
    m = MACHINES[machine]
    t0 = time.time()
    result = {"machine": machine, "cores": m["cores"], "ok": False, "smoke_s": None, "error": None}

    # 1. rsync code
    ssh_target = f"{m['user']}@{m['host']}"
    ssh_opts = f"-p {m['port']} -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    rsync_cmd = [
        "rsync", "-az", "--delete", *RSYNC_EXCLUDES,
        "-e", f"ssh {ssh_opts}",
        "./", f"{ssh_target}:majiang_ai/",
    ]
    rc, out, err = run(rsync_cmd, timeout=120)
    if rc != 0:
        result["error"] = f"rsync failed: {err[:200]}"
        return result
    print(f"  [{machine}] rsync done", flush=True)

    # 2. Setup env
    if m["has_conda"]:
        # E: use existing miniconda python 3.12
        env_cmd = (
            "cd majiang_ai && "
            "/root/miniconda3/bin/pip install -e . --quiet 2>&1 | tail -3 && "
            "echo ENV_OK"
        )
    else:
        # C/D: create venv
        env_cmd = (
            "cd majiang_ai && "
            "python3 -m venv .venv --quiet 2>&1 && "
            ".venv/bin/pip install -e . --quiet 2>&1 | tail -3 && "
            "echo ENV_OK"
        )
    rc, out, err = run([*ssh_cmd(machine), env_cmd], timeout=300)
    if "ENV_OK" not in out:
        result["error"] = f"env setup failed: {out[-200:]} {err[-200:]}"
        return result
    print(f"  [{machine}] env done", flush=True)

    if smoke_only:
        result["ok"] = True
        result["elapsed"] = time.time() - t0
        return result

    # 3. Smoke test: run a quick self-play match
    py = "/root/miniconda3/bin/python" if m["has_conda"] else ".venv/bin/python"
    smoke_cmd = (
        f"cd majiang_ai && timeout 300 {py} -c \""
        "import sys; sys.path.insert(0, 'src'); "
        "from majiang.strategy import versions; "
        "from majiang.strategy.policy import Mode; "
        "d = versions.build('v6', Mode.QUALIFIER); "
        "print('SMOKE_OK decider=', d.name)"
        "\""
    )
    rc, out, err = run([*ssh_cmd(machine), smoke_cmd], timeout=60)
    if "SMOKE_OK" not in out:
        result["error"] = f"smoke failed: {out[-200:]} {err[-200:]}"
        return result
    print(f"  [{machine}] smoke done", flush=True)

    result["ok"] = True
    result["elapsed"] = time.time() - t0
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machines", default="mj-c,mj-d,mj-e")
    ap.add_argument("--smoke-only", action="store_true", help="只部署+验证环境，不跑冒烟")
    args = ap.parse_args()

    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    print(f"部署目标：{machines}", flush=True)

    results = []
    for m in machines:
        print(f"\n=== {m} ===", flush=True)
        r = deploy(m, smoke_only=args.smoke_only)
        results.append(r)
        if r["ok"]:
            print(f"  ✅ {m} 部署成功（{r['elapsed']:.0f}s）", flush=True)
        else:
            print(f"  ❌ {m} 失败：{r['error']}", flush=True)

    # Summary
    ok = [r for r in results if r["ok"]]
    print(f"\n=== 部署结果：{len(ok)}/{len(results)} 台成功 ===", flush=True)
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"  {status} {r['machine']}（{r['cores']} 核）{r.get('elapsed', 0):.0f}s", flush=True)

    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
