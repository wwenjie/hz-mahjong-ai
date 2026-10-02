#!/usr/bin/env python3
"""只读值守：巡检比赛进程与日志新鲜度，异常写告警；对平台零请求，不启停任何进程。

设计约束（见 agent/AGENTS.md §3、agent/skills/mahjong-ops-watch/SKILL.md）：

1. **只读**。不改 `verify/**`、`scripts/**`，产物只落 `agent/out/`。
2. **零平台请求**。本文件不 import 任何网络库（`--audit` 会自我检查这一点）。
3. **不启停进程**。只用 `ps` 观测；重启由 `scripts/supervise.sh` 与 systemd 单元负责。
4. **同一根因只告警一次**，`--ack <key>` 后静默。
5. **终态不误报**：`scripts/supervise.sh` 记录「进程正常收工（赛事终态）」时，
   不报进程死亡，改出收尾摘要。
6. **状态落盘**，跨重启恢复计数——否则重启后计数器归零会变成静默偏置。

用法::

    uv run python agent/patrol/patrol.py --once      # 单轮，前台给结论
    uv run python agent/patrol/patrol.py             # 常驻循环
    uv run python agent/patrol/patrol.py --list-acks # 看已静默的根因
    uv run python agent/patrol/patrol.py --ack proc-down
    uv run python agent/patrol/patrol.py --audit     # 自检：无网络库、无进程启停
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "agent", "out")
LOG = os.path.join(OUT, "patrol.log")
ALERTS = os.path.join(OUT, "alerts.log")
STATUS = os.path.join(OUT, "patrol.status")
STATE = os.path.join(OUT, "patrol.state.json")

SUPERVISOR_LOG = os.path.join(ROOT, "logs", "supervisor.log")
WATCH_STATUS = os.path.join(ROOT, "verify", "out", "watch.status")
WATCH_INBOX = os.path.join(ROOT, "verify", "out", "inbox.log")

TERMINAL_MARK = "进程正常收工（赛事终态"
BREAKER_MARK = "判定为配置错误，守护停止"
FAST_FAIL_RE = re.compile(r"快速失败 (\d+)/(\d+)")

# supervisor.log 的“最后一段守护启动”超过这么多天，就认为其中的 breaker/fast_fail
# 是陈旧标记（不反映当前实况）。实测：09-24 的一次刻意熔断，到 09-28 仍被报成 MANUAL。
STALE_DAYS = 1.0

LEVELS = {"LOW": 1, "MED": 2, "HIGH": 3, "MANUAL": 4}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] {msg}\n")


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"alerts": {}, "logs": {}, "acked": []}


def save_state(st):
    os.makedirs(OUT, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=0)
    os.replace(tmp, STATE)


def emit(st, level, key, msg, extra=""):
    """同一根因只告警一次；人工 ack 后静默。"""
    if key in st.get("acked", []):
        return False
    rec = st.setdefault("alerts", {}).setdefault(key, {"count": 0, "level": level, "first": now()})
    if rec["count"] > 0 and LEVELS.get(level, 0) <= LEVELS.get(rec["level"], 0):
        rec["count"] += 1
        rec["last"] = now()
        return False
    rec.update({"count": rec["count"] + 1, "level": level, "last": now()})
    line = f"[{now()}] {level} {key} :: {msg}"
    if extra:
        line += f" | {extra}"
    os.makedirs(OUT, exist_ok=True)
    with open(ALERTS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    log(f"ALERT {line}")
    return True


# ---------- 巡检项 ----------

def find_processes(patterns):
    """返回 [(pid, lstart, cmd)]。先核对 pid 与启动时间，再判定（不 kill、不 pkill）。"""
    try:
        ps = subprocess.run(["ps", "-eo", "pid=,lstart=,args="],
                            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    hits = []
    for line in ps.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, lstart, cmd = parts[0], " ".join(parts[1:6]), parts[6]
        if "agent/patrol/patrol.py" in cmd:
            continue
        if any(p in cmd for p in patterns):
            hits.append((pid, lstart, cmd))
    return hits


def read_supervisor(sup_log):
    """解析 A 的守护日志：终态 / 熔断 / 当前快速失败计数。

    只看**最后一次「守护启动」之后**的行——否则历史熔断记录会被当成当前状态（实测误报过：
    2026-09-24 的一次熔断在三天后仍被判为「已熔断」）。
    """
    info = {"terminal": False, "breaker": False, "fast_fail": None, "tail": "",
            "last_start": None, "stale_days": None}
    try:
        with open(sup_log, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return info
    info["tail"] = "\n".join(lines[-8:])
    starts = [i for i, l in enumerate(lines) if "守护启动" in l]
    if not starts:
        return info
    seg = lines[starts[-1]:]
    info["last_start"] = seg[0].split(" ")[0]
    try:
        age = time.time() - time.mktime(time.strptime(info["last_start"][:19], "%Y-%m-%dT%H:%M:%S"))
        info["stale_days"] = round(age / 86400, 1)
    except ValueError:
        pass
    for line in seg:
        if TERMINAL_MARK in line:
            info["terminal"] = True
        if BREAKER_MARK in line:
            info["breaker"] = True
        m = FAST_FAIL_RE.search(line)
        if m:
            info["fast_fail"] = (int(m.group(1)), int(m.group(2)))
    # 陈旧标记：日志里最后一段“守护启动”距今超过 STALE_DAYS 天时，其中的
    # breaker / fast_fail 反映的**不是当前实况**（例如上一个采集时代遗留的熔断）。
    # 不区分会把 3 天前的熔断当成现在的事故报 MANUAL。
    info["stale"] = bool(info["stale_days"] and info["stale_days"] > STALE_DAYS)
    return info


def scan_logs(st, log_dir):
    """只读新增字节，统计事件增量 + 采集相决策耗时超阈。返回 (最新 mtime, 增量计数字典, 文件数)。

    decision.made 行逐条解析 elapsed_ms（A 16:10 派活：采集被压时会出现 >3s 卡顿，
    最坏实测 29.4s——29s 不是算贵是饿死/冻结，超 3s 即值得告警）。
    """
    delta = {"decision": 0, "error": 0, "timeout": 0}
    slow = []  # (elapsed_ms, ts, decider_tag)
    newest, nfiles = 0.0, 0
    seen = st.setdefault("logs", {})
    live = set()
    for path in glob.glob(os.path.join(log_dir, "*.jsonl")):
        nfiles += 1
        live.add(path)
        try:
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        newest = max(newest, mtime)
        rec = seen.get(path)
        if rec is None:                      # 新文件：从末尾起算，不把历史量算成本轮增量
            seen[path] = {"offset": size}
            continue
        offset = rec.get("offset", 0)
        if size < offset:                    # 被轮转/截断
            offset = 0
        if size > offset:
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read(size - offset).decode("utf-8", errors="replace")
            except OSError:
                continue
            for ev in ("error", "timeout"):
                delta[ev] += chunk.count(f'"event":"{ev}"') + chunk.count(f'"event": "{ev}"')
            for line in chunk.splitlines():
                if '"decision.made"' not in line:
                    continue
                delta["decision"] += 1
                try:
                    ev = json.loads(line)
                    ms = ev.get("elapsed_ms")
                    if isinstance(ms, (int, float)) and ms > 3000:
                        dec = str(ev.get("decider", ""))[:24]
                        slow.append((round(ms), ev.get("ts", "?"), dec))
                except (json.JSONDecodeError, AttributeError):
                    continue
            rec["offset"] = size
    for gone in set(seen) - live:
        seen.pop(gone, None)
    delta["slow"] = slow
    return newest, delta, nfiles


def read_watch_status():
    try:
        with open(WATCH_STATUS, encoding="utf-8") as f:
            return json.loads(f.read())
    except (OSError, json.JSONDecodeError):
        return {}


def new_inbox_lines(st):
    """B 的 inbox 新条目 → INFO 级告警（按内容去重）。"""
    fresh = []
    try:
        with open(WATCH_INBOX, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
    except OSError:
        return fresh
    seen = set(st.setdefault("inbox_seen", []))
    for line in lines:
        # 必须用稳定哈希：内建 hash() 每进程随机化，会让去重每轮失效（实测每轮重复告警）
        key = "inbox:" + hashlib.sha256(line.encode()).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        fresh.append((key, line))
    st["inbox_seen"] = sorted(seen)[-500:]
    return fresh


# ---------- 主循环 ----------

def one_cycle(st, args, verbose=True):
    summary = {}
    procs = find_processes(args.target)
    sup = read_supervisor(args.supervisor_log)
    newest, delta, nfiles = scan_logs(st, args.log_dir)
    fresh_sec = int(time.time() - newest) if newest else None

    # 1. 进程存活（终态时不报死亡）
    if sup["terminal"]:
        summary["process"] = "已进入赛事终态"
    elif procs:
        summary["process"] = f"存活 {len(procs)} 个"
    else:
        summary["process"] = "未发现"
        emit(st, "HIGH", "proc-down", "未发现目标进程",
             extra=f"supervisor 末行: {sup['tail'].splitlines()[-1] if sup['tail'] else 'n/a'}")

    # 2. 平台在线判据（用日志新鲜度作代理量，明确标注是代理）
    summary["log_fresh_sec"] = fresh_sec
    if not sup["terminal"]:
        if fresh_sec is None:
            emit(st, "MED", "log-missing", "logs/*.jsonl 不存在或为空")
        elif fresh_sec >= args.online_window:
            emit(st, "HIGH", "log-stale",
                 f"日志 {fresh_sec}s 未更新（在线判据窗口 {args.online_window}s，代理量）")

    # 3. 错误/超时速率（本轮增量）
    summary["delta"] = delta
    if delta["error"] > args.error_threshold:
        emit(st, "MED", "error-rate", f"本轮 error {delta['error']} 条（阈值 {args.error_threshold}）")
    if delta["timeout"] > args.timeout_threshold:
        emit(st, "MED", "timeout-rate",
             f"本轮 timeout {delta['timeout']} 条（阈值 {args.timeout_threshold}）")

    # 3b. 采集相决策卡顿（A 2026-10-02 16:10 派活）：decision.made elapsed_ms >3s。
    # 29s 级卡顿不是算贵是饿死/冻结（load 16-19/16 核压满）；>3s 即报、>10s 升级。
    slow = delta.get("slow") or []
    if slow:
        worst = max(slow)
        summary["slow_decisions"] = f"{len(slow)} 条, 最坏 {worst[0]}ms"
        level = "HIGH" if worst[0] > 10000 else "MED"
        detail = "; ".join(f"{ms}ms@{ts}[{dec}]" for ms, ts, dec in sorted(slow, reverse=True)[:5])
        emit(st, level, "decision-slow",
             f"采集相决策卡顿 {len(slow)} 条 >3s（最坏 {worst[0]}ms）——机器被压满的信号",
             extra=detail)

    # 4. 守护熔断
    if sup["breaker"] and not sup["stale"]:
        emit(st, "MANUAL", "breaker",
             "守护脚本已熔断，需人工介入（检查配置与令牌后手动重启）",
             extra=f"最后一次守护启动 {sup['last_start']}（{sup['stale_days']} 天前），"
                   f"末行: {sup['tail'].splitlines()[-1] if sup['tail'] else 'n/a'}")
    elif sup["breaker"] and sup["stale"]:
        emit(st, "LOW", "breaker-stale",
             f"supervisor.log 里的熔断标记已陈旧（最后启动 {sup['stale_days']} 天前），"
             "不代表当前实况，不告警",
             extra=f"最后启动 {sup['last_start']}")
    elif sup["fast_fail"]:
        n, m = sup["fast_fail"]
        summary["fast_fail"] = f"{n}/{m}"
        if n >= max(1, m - 1):
            emit(st, "HIGH", "breaker-near", f"快速失败 {n}/{m}，接近熔断")

    # 5. B 的守护心跳
    ws = read_watch_status()
    if ws:
        age = None
        try:
            age = int(time.time() - time.mktime(time.strptime(ws["alive_at"], "%Y-%m-%d %H:%M:%S")))
        except (KeyError, ValueError):
            pass
        summary["watch_age_sec"] = age
        if age is not None and age > args.watch_stale:
            emit(st, "LOW", "watch-stale", f"B 的 agent_watch 心跳 {age}s 未更新")

    # 6. B 的新条目
    for key, line in new_inbox_lines(st):
        emit(st, "LOW", key, f"B inbox 新条目: {line[:200]}")

    st["last_cycle"] = now()
    st["cycles"] = st.get("cycles", 0) + 1
    st.setdefault("counters", {})["events"] = delta
    save_state(st)

    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "alive_at": now(), "pid": os.getpid(), "cycles": st["cycles"],
            "process": summary.get("process"), "log_fresh_sec": fresh_sec,
            "log_files": nfiles, "delta": delta,
            "watch_age_sec": summary.get("watch_age_sec"),
            "breaker": sup["breaker"] and not sup["stale"],
            "breaker_stale": sup["stale"],
            "terminal": sup["terminal"],
        }, ensure_ascii=False) + "\n")

    if verbose:
        if sup["terminal"]:
            print("状态: 赛事终态（不报死亡告警）")
            print("收尾摘要: supervisor 记录 —— " +
                  (sup["tail"].splitlines()[-1] if sup["tail"] else "(无)"))
        else:
            print(f"状态: 进程 {summary.get('process')} / 日志新鲜度 {fresh_sec}s / "
                  f"文件 {nfiles} 个 / 本轮 decision={delta['decision']} "
                  f"error={delta['error']} timeout={delta['timeout']} / "
                  f"卡顿={summary.get('slow_decisions', '无')} / "
                  f"熔断={sup['breaker'] and not sup['stale']}"
                  f"{'（陈旧标记已忽略）' if sup['stale'] else ''} "
                  f"快速失败={summary.get('fast_fail', '-')} / "
                  f"B 心跳 {summary.get('watch_age_sec', '-')}s")
    return summary


def audit():
    """自检：用 AST 确认本文件无网络库、无进程启停调用（不做文本扫描，避免匹配到词表本身）。"""
    import ast

    path = os.path.abspath(__file__)
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    net_mods = {"urllib", "http", "requests", "socket", "ftplib", "telnetlib", "aiohttp", "ssl"}
    forbidden_calls = {("os", "system"), ("os", "popen"), ("os", "kill"), ("os", "killpg"),
                       ("subprocess", "Popen"), ("subprocess", "call"),
                       ("subprocess", "check_output")}
    allowed_cmds = {"ps"}

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in net_mods:
                    found.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in net_mods:
                found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                pair = (fn.value.id, fn.attr)
                if pair in forbidden_calls:
                    found.append(f"{pair[0]}.{pair[1]}()")
                elif pair == ("subprocess", "run"):
                    argv = node.args[0] if node.args else None
                    if isinstance(argv, (ast.List, ast.Tuple)) and argv.elts:
                        head = argv.elts[0]
                        if not (isinstance(head, ast.Constant) and head.value in allowed_cmds):
                            found.append(f"subprocess.run({ast.unparse(head)})")

    print(f"审计 {path}")
    if found:
        for item in found:
            print(f"  禁用形态: {item}")
        return 1
    print("  无网络库导入；唯一的子进程调用是 ps；无系统调用级进程启停")
    print("  结论：只读巡检，对平台零请求，不会启停任何进程")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="单轮巡检后退出")
    ap.add_argument("--interval", type=int, default=60, help="循环间隔秒（默认 60）")
    ap.add_argument("--target", action="append", default=None,
                    help="目标进程匹配串，可多次（默认 python -m majiang）")
    ap.add_argument("--online-window", type=int, default=90, help="平台在线判据窗口秒（默认 90）")
    ap.add_argument("--error-threshold", type=int, default=5)
    ap.add_argument("--timeout-threshold", type=int, default=50)
    ap.add_argument("--watch-stale", type=int, default=900, help="B 心跳过期阈值秒")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "logs"),
                    help="JSONL 日志目录（默认 <repo>/logs；测试场景可指向临时目录）")
    ap.add_argument("--supervisor-log", default=SUPERVISOR_LOG,
                    help="守护日志路径（默认 <repo>/logs/supervisor.log）")
    ap.add_argument("--ack", action="append", default=None, help="静默某个根因键")
    ap.add_argument("--list-acks", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    args.target = args.target or ["auto_session.py", "python -m majiang"]

    if args.audit:
        sys.exit(audit())

    st = load_state()
    if args.ack:
        st.setdefault("acked", [])
        for k in args.ack:
            if k not in st["acked"]:
                st["acked"].append(k)
        save_state(st)
        print("已静默: " + ", ".join(args.ack))
        return
    if args.list_acks:
        acks = st.get("acked", [])
        print("已静默根因: " + (", ".join(acks) if acks else "(无)"))
        return

    os.makedirs(OUT, exist_ok=True)
    log(f"patrol 启动 pid={os.getpid()} once={args.once} interval={args.interval} "
        f"target={args.target}")
    if args.once:
        one_cycle(st, args)
        return
    while True:
        try:
            one_cycle(st, args, verbose=False)
        except Exception as exc:                       # 巡检不许自己死掉
            log(f"巡检轮次异常（已记录，下轮继续）: {type(exc).__name__}: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
