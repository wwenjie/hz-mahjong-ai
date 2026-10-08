"""远端机操作助手（paramiko）——**密码只从环境变量读，绝不入仓**。

用法::

    export MAJIANG_REMOTE_PW='...'          # 必需（或 --password-file 指向 600 权限文件）
    .venv/bin/python tools/remote.py survey
    .venv/bin/python tools/remote.py exec "nproc; df -h /root | tail -1"
    .venv/bin/python tools/remote.py put  data/x.json  /root/majiang_ai/data/x.json
    .venv/bin/python tools/remote.py push-tree . /root/majiang_ai --exclude .venv --exclude data --exclude logs --exclude runs --exclude .git
    .venv/bin/python tools/remote.py get  /root/majiang_ai/out.json  agent/out/out.json

主机/端口/用户可用 `--host/--port/--user` 或环境变量 `MAJIANG_REMOTE_HOST/PORT/USER` 覆盖。
默认: `connect.nma1.seetacloud.com:53838 root`。
"""
from __future__ import annotations

import argparse
import os
import posixpath
import subprocess
import sys
from pathlib import Path

import paramiko

DEFAULT_HOST = os.environ.get("MAJIANG_REMOTE_HOST", "connect.nma1.seetacloud.com")
DEFAULT_PORT = int(os.environ.get("MAJIANG_REMOTE_PORT", "53838"))
DEFAULT_USER = os.environ.get("MAJIANG_REMOTE_USER", "root")


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get("MAJIANG_REMOTE_PW")
    if not password and args.password_file:
        password = Path(args.password_file).read_text(encoding="utf-8").strip()
    if not password:
        raise SystemExit("缺少密码：请 `export MAJIANG_REMOTE_PW=...` 或给 --password-file（权限 600）")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=25,
        banner_timeout=25,
        auth_timeout=25,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: float = 120.0, quiet: bool = False) -> int:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=False)
    stdin.close()
    for line in stdout:
        if not quiet:
            sys.stdout.write(line)
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if err and not quiet:
        sys.stdout.write(err)
    return code


def push_tree(client: paramiko.SSHClient, local: str, remote: str, excludes: list[str],
              compress: bool = True, progress_mb: int = 200) -> None:
    """把本地目录经**流式 tar** 推到远端并解开（不落盘中转、不把整包读进内存）。

    本地只做 tar（必要时 gzip），远端只做解包；两边都不需要 rsync/sshpass。
    """
    root = Path(local).resolve()
    names = [
        str(path.relative_to(root))
        for path in sorted(root.rglob("*"))
        if not path.is_dir() and not any(part in excludes for part in path.relative_to(root).parts)
    ]
    if not names:
        print("  无可推送文件")
        return
    flag = "-czf" if compress else "-cf"
    proc = subprocess.Popen(
        ["tar", flag, "-", "-C", str(root), *names],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    run(client, f"mkdir -p {remote}")
    stdin, stdout, stderr = client.exec_command(
        f"tar {'-xzf' if compress else '-xf'} - -C {remote}", timeout=7200
    )
    total = 0
    mark = progress_mb * 1_000_000
    next_mark = mark
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(1 << 20)
        if not chunk:
            break
        stdin.write(chunk)
        total += len(chunk)
        if total >= next_mark:
            print(f"  已推送 {total / 1e6:.0f} MB ...", flush=True)
            next_mark += mark
    proc.stdout.close()
    proc.wait()
    stdin.channel.shutdown_write()
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    print(f"  推送完成：{len(names)} 个文件 / {total / 1e6:.1f} MB，远端解包 exit={code} {err[:200]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="远端机操作（paramiko）")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--password-file", default=os.environ.get("MAJIANG_REMOTE_PW_FILE", ""))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("survey")
    p_exec = sub.add_parser("exec")
    p_exec.add_argument("command")
    p_exec.add_argument("--timeout", type=float, default=120.0)
    p_put = sub.add_parser("put")
    p_put.add_argument("local")
    p_put.add_argument("remote")
    p_get = sub.add_parser("get")
    p_get.add_argument("remote")
    p_get.add_argument("local")
    p_tree = sub.add_parser("push-tree")
    p_tree.add_argument("local")
    p_tree.add_argument("remote")
    p_tree.add_argument("--exclude", action="append", default=[])
    p_tree.add_argument("--no-compress", action="store_true")
    args = ap.parse_args(argv)

    client = connect(args)
    try:
        if args.cmd == "survey":
            commands = [
                "hostname; uname -a | cut -c1-80",
                "nproc; free -g | head -2; df -h /root / 2>/dev/null | tail -2",
                "python3 -V; which uv rsync tmux git 2>/dev/null",
                "nvidia-smi -L 2>/dev/null | head -2 || echo '(无 GPU 或 nvidia-smi 缺失)'",
                "ls /root 2>/dev/null | head -20",
                "ls -d /root/majiang_ai /root/majiang_rl /root/majiang_nnrl 2>/dev/null",
                "ps -eo pid,etime,cmd --sort=-etime 2>/dev/null | grep -E 'python|ab_test|jupyter' | grep -v grep | head -8",
            ]
            for command in commands:
                print(f"\n$ {command}")
                run(client, command, timeout=60)
            return 0
        if args.cmd == "exec":
            return run(client, args.command, timeout=args.timeout)
        if args.cmd == "put":
            sftp = client.open_sftp()
            run(client, f"mkdir -p {posixpath.dirname(args.remote)}")
            sftp.put(args.local, args.remote)
            print(f"  已上传 {args.local} → {args.remote}")
            return 0
        if args.cmd == "get":
            sftp = client.open_sftp()
            Path(args.local).parent.mkdir(parents=True, exist_ok=True)
            sftp.get(args.remote, args.local)
            print(f"  已下载 {args.remote} → {args.local}")
            return 0
        if args.cmd == "push-tree":
            push_tree(client, args.local, args.remote, args.exclude,
                      compress=not args.no_compress)
            return 0
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
