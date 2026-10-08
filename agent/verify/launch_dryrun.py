#!/usr/bin/env python3
"""B2：`python -m majiang` 端到端启动干跑（离线，**零平台请求**）。

做法：每个场景都在子进程里跑 `python -m majiang ...`，子进程的 `PYTHONPATH` 前置一个
临时目录，里面放 `sitecustomize.py` 把 `socket.socket` / `create_connection` / `getaddrinfo` /
`gethostbyname` 替换成「记录 + 抛错」。于是：

* 任何一次网络尝试都会写进 netlog，**且**让该子进程报错 → 要么「零请求」要么「当场现形」，
  不存在「悄悄连上了」的灰色地带。
* 场景自带**正对照**（假令牌 + 不可达地址）：它必须失败并留下 NETWORK_ATTEMPT，
  否则说明阻断器没生效，全部结论作废。

只读 src/、零平台请求、不写任何 src/ 文件。产物：`agent/out/launch-dryrun.log`。

用法：`uv run python agent/verify/launch_dryrun.py`
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "agent" / "out" / "launch-dryrun.log"

# 子进程里生效的「离线闸门」：任何 socket 动作都记一笔再抛错。
SITECUSTOMIZE = textwrap.dedent(
    """
    import os, socket

    _LOG = os.environ["MAJIANG_NETLOG"]


    def _deny(*_a, **_k):
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write("NETWORK_ATTEMPT\\n")
        raise RuntimeError("离线干跑：网络被阻断")


    class _BlockedSocket:
        def __init__(self, *_a, **_k):
            _deny()


    socket.socket = _BlockedSocket
    socket.create_connection = _deny
    socket.getaddrinfo = _deny
    socket.gethostbyname = _deny
    """
)

# (名字, argv, 期望退出码或 None=非零即可, 期望在 stderr/stdout 里出现的片段,
#  want_net: "zero"=必须零网络尝试；"at_least_one"=正对照，必须被闸门拦下并记账)
SCENARIOS: list[tuple[str, list[str], int | None, str, str]] = [
    ("help", ["--help"], 0, "杭州麻将 AI 参赛程序", "zero"),
    ("no-token", [], 1, "必须提供", "zero"),
    ("missing-token-env", ["--token-env", "MAJIANG_NO_SUCH_TOKEN_B2"], 1, "为空", "zero"),
    ("unknown-decider", ["--token-env", "MAJIANG_FAKE_TOKEN_B2", "--decider", "__nope__"], 1, "未知决策器", "zero"),
    ("bad-mode", ["--mode", "__nope__"], 2, "invalid choice", "zero"),
    (
        "positive-control",
        [
            "--token-env",
            "MAJIANG_FAKE_TOKEN_B2",
            "--skip-version-check",
            "--server",
            "http://127.0.0.1:9",
            "--duration",
            "1",
            "--log-dir",
            "{tmp}",
        ],
        None,
        "",
        "at_least_one",
    ),
]


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="majiang-dryrun-"))
    gate = tmp / "gate"
    gate.mkdir()
    (gate / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    netlog = tmp / "netlog.txt"

    lines: list[str] = []
    failures: list[str] = []
    blocked_total = 0

    for name, argv, want_code, want_text, want_net in SCENARIOS:
        netlog.write_text("", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(gate) + os.pathsep + env.get("PYTHONPATH", "")
        env["MAJIANG_NETLOG"] = str(netlog)
        env["MAJIANG_FAKE_TOKEN_B2"] = "tok-b2-fake"
        argv = [a.replace("{tmp}", str(tmp / "logs")) for a in argv]
        proc = subprocess.run(
            [sys.executable, "-m", "majiang", *argv],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        blocked = netlog.read_text(encoding="utf-8").count("NETWORK_ATTEMPT")
        blocked_total += blocked
        blob = proc.stdout + proc.stderr
        code_ok = (proc.returncode == want_code) if want_code is not None else (proc.returncode != 0)
        text_ok = want_text in blob if want_text else True
        net_ok = blocked == 0 if want_net == "zero" else blocked > 0
        ok = code_ok and text_ok and net_ok
        if not ok:
            failures.append(name)
        lines.append(
            json.dumps(
                {
                    "scenario": name,
                    "argv": argv,
                    "returncode": proc.returncode,
                    "expected_code": want_code,
                    "network_attempts": blocked,
                    "want_net": want_net,
                    "code_ok": code_ok,
                    "text_ok": text_ok,
                    "net_ok": net_ok,
                    "verdict": "PASS" if ok else "FAIL",
                    "stdout_first_line": (proc.stdout.strip().splitlines() or [""])[0][:160],
                    "stderr_first_line": (proc.stderr.strip().splitlines() or [""])[0][:160],
                },
                ensure_ascii=False,
            )
        )

    # 正对照必须自己留下一条网络尝试，否则说明闸门没挂上，全部结论作废。
    control_ok = "positive-control" not in failures
    verdict = "PASS" if not failures and control_ok and blocked_total > 0 else "FAIL"

    report = {
        "verdict": verdict,
        "scenarios_run": len(SCENARIOS),
        "failures": failures,
        "total_network_attempts_in_negative_scenarios": blocked_total
        - (1 if "positive-control" not in failures else 0),
        "positive_control_network_attempts": blocked_total,
        "note": "负向场景期望 network_attempts==0；正对照必须 >0（证明闸门生效）。",
    }

    OUT.write_text(
        "\n".join(
            [
                "# B2 `python -m majiang` 离线启动干跑",
                f"# repo={REPO} python={sys.version.split()[0]}",
                "# 闸门：sitecustomize 把 socket.socket/create_connection/getaddrinfo/gethostbyname 全换成抛错+记账",
                "",
                *lines,
                "",
                json.dumps(report, ensure_ascii=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"log -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
