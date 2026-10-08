#!/usr/bin/env python3
"""SSH 密码交互登录执行远程命令（无 sshpass/expect 环境的替代品）。

用法: python tools/ssh_run.py <host> <port> <user> <password> <remote_command>
"""
import os
import pty
import select
import sys
import time


def ssh_run(host: str, port: int, user: str, password: str, cmd: str, timeout: int = 60) -> int:
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "PreferredAuthentications=password",
        "-o", "PubkeyAuthentication=no",
        "-o", "NumberOfPasswordPrompts=1",
        "-o", "ConnectTimeout=15",
        "-p", str(port),
        f"{user}@{host}",
        cmd,
    ]
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("ssh", ssh_cmd)
        os._exit(127)

    output = b""
    sent_pwd = False
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 2)
        if fd in r:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
            if not sent_pwd and b"assword" in output:
                os.write(fd, password.encode() + b"\n")
                sent_pwd = True
        # check child
        wpid, sts = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            status = os.WEXITSTATUS(sts) if os.WIFEXITED(sts) else -1
            # drain remaining output
            try:
                while True:
                    r2, _, _ = select.select([fd], [], [], 0.5)
                    if fd in r2:
                        chunk = os.read(fd, 4096)
                        if not chunk:
                            break
                        output += chunk
                    else:
                        break
            except OSError:
                pass
            break
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        print("TIMEOUT", file=sys.stderr)
        return 124

    os.close(fd)
    sys.stdout.write(output.decode("utf-8", errors="replace"))
    return status if status is not None else 1


if __name__ == "__main__":
    host, port, user, password = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    cmd = sys.argv[5] if len(sys.argv) > 5 else "echo LOGIN_OK; nproc; free -m | head -2; df -h / | tail -1; uname -a"
    sys.exit(ssh_run(host, port, user, password, cmd))
