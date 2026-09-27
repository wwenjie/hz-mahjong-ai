"""离线迭代循环：跑预先登记的配对 A/B 队列，并刷新状态页。

**为什么是这个形态**：守护进程不能替我判断「下一个该试什么」——那是推理，不是定时任务。
所以把可自动化的部分切出来：**我预先登记候选档位与假设，它替我跑完配对自对弈并写结果**。
下次接手时读 `data/experiments/results.jsonl` 即可，不必等人在场。

三条安全约束：

1. **不碰平台**。只用 `tools/ab_test.py` 的配对自对弈（同副牌、四座位旋转）。
   真机实验必须人工决定——令牌限速按用户 16/s，多一个采集器就会互相挤兑。
2. **子进程降优先级**（`nice -n 15`），避免把真机采集的决策窗口挤到超时。
3. **单实例锁 + 队列跑空即退出**。不做无限循环：无人值守的无限循环只会积累风险。

它只**报告**，从不改默认档。是否采纳由人决定——这一条是刻意的：
自对弈已被证明可能给出与真机不一致的方向。

用法::

    tools/queue_supervisor.sh            # 带重启的守护（推荐）
    uv run python tools/iterate_loop.py  # 直接跑一轮
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 队列放在 notes/ 而不是 data/：**data/ 是 gitignore 的**，而「预登记」的价值就在于
# 有版本记录（什么时候登记了什么假设、结果如何）。放 data/ 会导致它不入库、B 也看不到。
QUEUE = Path("notes/experiments.json")
LOCK = Path("data/experiments/.lock")
STATUS = Path("notes/STATUS.md")
THREAD = Path("notes/THREAD.md")

METRIC_RE = re.compile(
    r"^\s*(?P<name>\S+)\s+均值\s+(?P<mean>[-+]?[\d.]+)\s+标准误\s+(?P<se>[\d.]+)"
    r"\s+t\s+(?P<t>[-+]?[\d.]+)"
)
INTERESTING = ("总得分", "名次分", "白板数", "胡次数", "番数总和")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_queue() -> dict:
    if not QUEUE.exists():
        return {"jobs": []}
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def save_queue(queue: dict) -> None:
    QUEUE.write_text(json.dumps(queue, ensure_ascii=False, indent=1), encoding="utf-8")


def parse_ab_output(text: str) -> dict:
    """从 ab_test 的 stdout 抽五个指标的 (均值, 标准误, t)。"""
    metrics: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match and match["name"] in INTERESTING:
            metrics[match["name"]] = {
                "mean": float(match["mean"]),
                "se": float(match["se"]),
                "t": float(match["t"]),
            }
    return metrics


def run_job(job: dict, timeout_sec: float) -> dict:
    """跑一个 job（可能含多个种子），返回汇总结果。"""
    treatment = job["treatment"]
    baseline = job.get("baseline", "heuristic")
    matches = int(job.get("matches", 300))
    seeds = job.get("seeds") or [20260926]
    field = job.get("field") or ""
    pooled: dict[str, list[tuple[float, float]]] = {}
    raw: list[str] = []
    for seed in seeds:
        cmd = [
            "nice", "-n", "15",
            "uv", "run", "python", "tools/ab_test.py",
            "--treatment", treatment,
            "--baseline", baseline,
            "--matches", str(matches),
            "--seed", str(seed),
        ]
        if field:
            # 换掉「另三座坐谁」。这是评估副露类假设的必要条件：默认 field=baseline
            # 等于让我们对着三个几乎不副露的复制品打分，结构上测不出副露的价值。
            cmd += ["--field", field]
        proc = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout_sec
        )
        raw.append(proc.stdout[-4000:])
        if proc.returncode != 0:
            return {
                "status": "failed",
                "detail": f"ab_test 退出码 {proc.returncode}: {proc.stderr[-500:]}",
            }
        metrics = parse_ab_output(proc.stdout)
        if not metrics:
            return {"status": "failed", "detail": "未能从输出解析出指标"}
        for name, values in metrics.items():
            pooled.setdefault(name, []).append((values["mean"], values["se"]))

    # 多种子按逆方差加权合并（标准误越小权重越大）
    combined = {}
    for name, pairs in pooled.items():
        weight_sum = sum(1.0 / (se**2) for _, se in pairs if se > 0)
        if weight_sum <= 0:
            continue
        mean = sum(m / (se**2) for m, se in pairs if se > 0) / weight_sum
        se = (1.0 / weight_sum) ** 0.5
        combined[name] = {"mean": mean, "se": se, "t": mean / se if se else 0.0}
    return {"status": "done", "seeds": seeds, "matches": matches, "metrics": combined}


def refresh_status(queue: dict, note: str = "") -> None:
    """写状态页：一眼看完「数据多少、在跑什么、B 有没有留言」。"""
    lines = [f"# 状态（{now_iso()}）", ""]
    if note:
        lines += [note, ""]

    events = len(list(Path("data/auto_sessions").glob("*/events/*.json")))
    rooms = len([p for p in Path("data/auto_sessions").iterdir() if p.is_dir()])
    sessions = 0
    ledger = Path("data/auto_sessions/sessions.jsonl")
    if ledger.exists():
        sessions = sum(1 for line in ledger.read_text(errors="replace").splitlines() if line.strip())
    lines += [
        "## 数据",
        f"- 事件流 {events} 个 / 房 {rooms} 个 / 会话 {sessions} 场",
        "",
    ]

    lines += ["## 实验队列"]
    jobs = queue.get("jobs", [])
    if not jobs:
        lines.append("- （空）")
    for job in jobs:
        mark = {"pending": "待跑", "running": "在跑", "done": "完成", "failed": "失败"}.get(
            job.get("status", "pending"), "?"
        )
        lines.append(
            f"- [{mark}] `{job['id']}` {job['treatment']} vs {job.get('baseline', 'heuristic')}"
            f" ×{job.get('matches', 300)} 场/{len(job.get('seeds') or [0])} 种子"
        )
        if job.get("hypothesis"):
            lines.append(f"  - 假设：{job['hypothesis']}")
        for name, value in (job.get("result") or {}).get("metrics", {}).items():
            lines.append(
                f"  - {name}: {value['mean']:+.3f}（t {value['t']:+.2f}）"
            )
    lines += [""]

    lines += ["## 消息（notes/THREAD.md）"]
    if THREAD.exists():
        entries = [line for line in THREAD.read_text(encoding="utf-8").splitlines() if line.startswith("### ")]
        lines.append(f"- 共 {len(entries)} 条")
        for entry in entries[-4:]:
            lines.append(f"  - {entry[4:]}")
    else:
        lines.append("- （无）")
    lines += [""]

    try:
        head = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad %s", "--date=format:%m-%d %H:%M"],
            cwd=REPO, capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        lines += ["## 仓库", f"- 最新提交 {head}", ""]
    except Exception:  # noqa: BLE001
        pass

    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def acquire_lock() -> bool:
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
            os.kill(pid, 0)
            print(f"已有实例在跑（pid {pid}），退出")
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线实验队列循环")
    parser.add_argument(
        "--timeout",
        type=float,
        default=14400,
        help=(
            "单个 job 的秒级上限。**默认值依据实测**：在三个守护共存、ab_test 被 nice 的"
            "情况下约 3.5 秒/场（不是理论上的 1.6 秒——CPU 争用很真实）。"
            "所以 300 场 × 4 座位旋转 ≈ 70 分钟/种子。原先默认 7200（2 小时）会让"
            "「300 场 × 2 种子」的 job 全部超时作废——曾因此白跑 15 小时。"
        ),
    )
    parser.add_argument("--loop", action="store_true", help="跑空后不退出，等待新 job 加入")
    parser.add_argument("--poll", type=float, default=300, help="--loop 下的轮询间隔")
    args = parser.parse_args(argv)

    if not acquire_lock():
        return 1
    try:
        while True:
            queue = load_queue()
            pending = [j for j in queue.get("jobs", []) if j.get("status") in (None, "pending")]
            refresh_status(queue, note=f"待跑 {len(pending)} 个" if pending else "队列已跑空")
            if not pending:
                if not args.loop:
                    print("队列已跑空，退出")
                    return 0
                time.sleep(args.poll)
                continue

            job = pending[0]
            job["status"] = "running"
            job["started_at"] = now_iso()
            save_queue(queue)
            refresh_status(load_queue(), note=f"在跑 `{job['id']}`")
            print(f"[{now_iso()}] 开始 {job['id']}: {job['treatment']}")
            try:
                result = run_job(job, args.timeout)
            except subprocess.TimeoutExpired:
                result = {
                    "status": "failed",
                    "detail": f"超时（>{args.timeout}s）。按实测约 3.5 秒/场估算预算："
                    f"matches × 4 旋转 × 种子数 × 3.5 秒，别超过 --timeout",
                }
            except Exception as exc:  # noqa: BLE001
                result = {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}

            job["result"] = result
            job["status"] = result["status"]
            job["ended_at"] = now_iso()
            queue = load_queue()
            for index, existing in enumerate(queue.get("jobs", [])):
                if existing.get("id") == job["id"]:
                    queue["jobs"][index] = job
            save_queue(queue)
            print(f"[{now_iso()}] {job['id']} -> {result['status']}")
    finally:
        LOCK.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
