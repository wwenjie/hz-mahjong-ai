#!/usr/bin/env python3
"""B 的守护进程：watch + verify + track（不打平台、不改任何人的文件）。

每 POLL_SEC 秒一轮：
1. watch  — 检测 notes/agent-a.md 新条目（含 `### 答 B:`/`### 问 B:`），摘要记入 inbox
2. verify — data/auto_sessions 文件数增长 ≥GROWTH_THRESHOLD 时重跑不变量校验，
            违反数 >0 立刻写进 inbox（早期抓数据污染）
3. track  — 跟踪冻结清单之后的新增 finished 场数；达到 REANALYZE_AT 场提醒做独立复算

产物（全部在 verify/out/，B 的地盘）：
  watch.log      滚动日志
  inbox.log      给 B 的待办队列（A 的新条目摘要、数据警报）
  watch.status   一行心跳（last_seen 时间戳 + 计数），供人/进程确认存活
  watch.state.json  跨重启的持久状态（文件哈希、已见条目、上次校验的文件数）
  watch.pid      自身 pid（只读不写别的进程）

启动（经 supervisor，含 setsid 脱离会话）：
  nohup setsid bash scripts/agent_watch_supervisor.sh >/dev/null 2>&1 &
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "verify", "out")
STATE = os.path.join(OUT, "watch.state.json")
LOG = os.path.join(OUT, "watch.log")
INBOX = os.path.join(OUT, "inbox.log")
STATUS = os.path.join(OUT, "watch.status")
PIDFILE = os.path.join(OUT, "watch.pid")
AGENT_A = os.path.join(ROOT, "notes", "agent-a.md")
THREAD = os.path.join(ROOT, "notes", "THREAD.md")
MANIFEST = os.path.join(ROOT, "notes", "manifest-20260926.txt")
INVARIANTS = os.path.join(ROOT, "verify", "invariants.py")

POLL_SEC = 120
GROWTH_THRESHOLD = 50      # 新增 ≥50 份事件文件才重跑不变量（全量一遍约 40s）
REANALYZE_AT = 200         # 新增 ≥200 个 finished 场提醒独立复算


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def inbox(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    log(f"INBOX: {msg}")


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, STATE)


def file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def new_entries(path, seen_hashes):
    """返回 (新增 ## 条目标题列表, 给 B 的段落摘要, 更新后的 seen)。"""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return [], [], seen_hashes
    entries = []
    current = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                entries.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append("\n".join(current))
    fresh_titles, for_b = [], []
    now_seen = set(seen_hashes)
    for entry in entries:
        h = hashlib.sha256(entry.encode()).hexdigest()[:16]
        if h in now_seen:
            continue
        now_seen.add(h)
        title = entry.splitlines()[0][3:].strip()
        fresh_titles.append(title)
        for para in entry.splitlines():
            if para.startswith("### 答 B") or para.startswith("### 问 B"):
                for_b.append(para.strip())
    return fresh_titles, for_b, sorted(now_seen)


def count_event_files():
    total = 0
    base = os.path.join(ROOT, "data", "auto_sessions")
    try:
        for room in os.listdir(base):
            evdir = os.path.join(base, room, "events")
            if os.path.isdir(evdir):
                total += sum(1 for f in os.listdir(evdir) if f.endswith(".json"))
    except OSError:
        pass
    return total


def run_invariants():
    """重跑不变量校验；返回 (是否通过, 摘要行)。nice -n 15（算力纪律 §5.4）。"""
    try:
        proc = subprocess.run(
            ["nice", "-n", "15", sys.executable, INVARIANTS],
            cwd=ROOT, capture_output=True, text=True, timeout=900,
        )
        first = (proc.stdout or "").splitlines()
        summary = " | ".join(first[:3]) if first else "(无输出)"
        return proc.returncode == 0, summary
    except Exception as exc:  # noqa: BLE001 —— 守护进程不许死
        return False, f"invariants 运行异常: {type(exc).__name__}: {exc}"


def heartbeat(st):
    try:
        with open(STATUS, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "alive_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pid": os.getpid(),
                "files_seen": st.get("files_seen", 0),
                "entries_seen": len(st.get("entries", [])),
                "invariants_last_ok": st.get("invariants_ok"),
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def already_running():
    try:
        with open(PIDFILE, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"agent_watch" in f.read()
    except OSError:
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    if already_running():
        print("agent_watch 已在运行，退出", file=sys.stderr)
        sys.exit(1)
    with open(PIDFILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    st = load_state()
    st.setdefault("entries", [])
    st.setdefault("files_seen", 0)
    log(f"agent_watch 启动 pid={os.getpid()} poll={POLL_SEC}s")
    inbox("agent_watch 上线：watch(agent-a.md) + verify(invariants) + track(新数据)")

    while True:
        try:
            # 1. watch A 的条目
            h = file_hash(AGENT_A)
            if h and h != st.get("agent_a_hash"):
                titles, for_b, st["entries"] = new_entries(AGENT_A, set(st["entries"]))
                st["agent_a_hash"] = h
                if titles:
                    inbox(f"A 新增 {len(titles)} 条: {'; '.join(titles)}")
                for line in for_b:
                    inbox(f"A→B: {line}")

            # 1b. watch THREAD.md 里 TO B 的新消息（协议 §3 通道）
            ht = file_hash(THREAD)
            if ht and ht != st.get("thread_hash"):
                st["thread_hash"] = ht
                try:
                    with open(THREAD, encoding="utf-8") as f:
                        thread_text = f.read()
                    msgs = [m for m in thread_text.split("\n### ")
                            if " TO B " in m.splitlines()[0]]  # 只看标题行，防正文误命中
                    seen_t = set(st.get("thread_msgs", []))
                    for m in msgs:
                        mh = hashlib.sha256(m.encode()).hexdigest()[:16]
                        if mh not in seen_t:
                            seen_t.add(mh)
                            inbox(f"THREAD TO B 新消息: {m.splitlines()[0].strip()}")
                    st["thread_msgs"] = sorted(seen_t)
                except OSError:
                    pass

            # 2./3. 数据增长 → 重校验 + 复算提醒
            # 与「上次已校验的文件数」比（不是上一轮轮询——否则每轮只增 1~2 永远不触发）
            files = count_event_files()
            verified = st.get("files_verified", 0)
            if files and verified and files - verified >= GROWTH_THRESHOLD:
                ok, summary = run_invariants()
                st["invariants_ok"] = ok
                if ok:
                    log(f"数据 {verified}→{files}，不变量重跑通过: {summary}")
                else:
                    inbox(f"!! 数据 {verified}→{files}，不变量校验失败: {summary}")
                st["files_verified"] = files
                new_matches = files - 1129  # 冻结清单基线
                if new_matches >= REANALYZE_AT and not st.get("reanalyze_flagged"):
                    inbox(f"清单外新增 {new_matches} 份事件文件，"
                          f"已够独立复算 A/B 的样本量（阈值 {REANALYZE_AT}）")
                    st["reanalyze_flagged"] = True
            if files:
                st["files_seen"] = files
                if not verified:
                    st["files_verified"] = files  # 首轮基线

            save_state(st)
            heartbeat(st)
        except Exception:
            log("轮询异常（已吞，下轮继续）:\n" + traceback.format_exc())
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
