"""从 `data/experiments/logs/` 的原始 stdout 恢复 job 的完成记录。

**为什么需要它**：`notes/experiments.json` 由 `iterate_loop.py` 持续写入，但**不是每次写入都提交**。
2026-10-03 我用 `git checkout -- notes/experiments.json` 清 diff 时，把 **22 条已完成的记录**一起
回滚成了 `pending`（HEAD 停在它们跑完之前），于是 `iterate_loop` 开始**重复计算**它们——
而单 job 实测要 **5544s（1.5h）**，21 条重跑会白占约 16 小时墙钟，还会挡住后面的决策臂。

**恢复是安全的**：`iterate_loop.run_job` 把每个种子的**完整 stdout 和命令行**都写进了
`logs/<job-id>-seed<seed>.log`，而指标就是从 stdout 用 `parse_ab_output` 解析出来的。
所以这里**复用同一个解析函数与同一个合并口径**（多种子按逆方差加权），
得到的结果与当初跑完写下的**同一份**，不是重新估计。

用法::

    uv run python tools/recover_jobs_from_logs.py            # 只体检，不改文件
    uv run python tools/recover_jobs_from_logs.py --apply    # 写回完成记录
    uv run python tools/recover_jobs_from_logs.py --backfill-missing --apply
        # 只给**已完成**的 job 补上「日志里有、result 里没有」的指标键。

**为什么需要 `--backfill-missing`**（2026-10-03 17:xx 的一次真事故）：`iterate_loop` 是**长跑守护**，
它的 `parse_ab_output` 用的是**自己启动时导入的** `INTERESTING` 元组。当天我在守护运行中给
`ab_test.py` 加了新指标「每场名次分」并把名字加进 `INTERESTING`——于是**子进程的 stdout 里有这一项，
但守护的解析器把它丢掉**，`result.metrics` 里就少了。若不处理，8 条刚入队的重跑会**全部白跑**。
回填的做法是重新解析同一份 stdout（同一个 `parse_ab_output`，此时是新元组），
并且**只补缺失的键、绝不覆盖已有键**——已有键是当初写下的原始记录，不能被事后重算替换。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.iterate_loop import INTERESTING, parse_ab_output  # noqa: E402

QUEUE = REPO / "notes/experiments.json"
LOGS = REPO / "data/experiments/logs"


def parse_all_seeds(job: dict) -> dict | None:
    """所有种子都有**完整**日志时返回合并后的 metrics，否则 None。"""
    seeds = job.get("seeds") or [20260926]
    pooled: dict[str, list[tuple[float, float]]] = {}
    for seed in seeds:
        path = LOGS / f"{job['id']}-seed{seed}.log"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        body = text.split("[stderr]", 1)[0]
        # 完整性判据：ab_test 结尾那行「配对样本 N 场」必须在——半截 stdout 不算数
        if "配对样本" not in body:
            return None
        metrics = parse_ab_output(body)
        if not metrics:
            return None
        for name, values in metrics.items():
            pooled.setdefault(name, []).append((values["mean"], values["se"]))
    if "总得分" not in pooled or "名次分" not in pooled:
        return None
    combined: dict[str, dict[str, float]] = {}
    for name, pairs in pooled.items():
        weight_sum = sum(1.0 / (se**2) for _, se in pairs if se > 0)
        if weight_sum <= 0:
            continue
        mean = sum(m / (se**2) for m, se in pairs if se > 0) / weight_sum
        se = (1.0 / weight_sum) ** 0.5
        combined[name] = {"mean": mean, "se": se, "t": mean / se if se else 0.0}
    return {name: combined[name] for name in INTERESTING if name in combined}


def recoverable(job: dict) -> dict | None:
    """兼容旧名（`--apply` 分支用）。"""
    return parse_all_seeds(job)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 job 日志恢复完成记录")
    ap.add_argument("--apply", action="store_true", help="写回 experiments.json（默认只体检）")
    ap.add_argument("--only", default="", help="只处理 id 含该子串的 job")
    ap.add_argument(
        "--backfill-missing",
        action="store_true",
        help="只给**已完成**的 job 补「日志里有、result 里没有」的指标键（不覆盖已有键）",
    )
    args = ap.parse_args(argv)

    doc = json.loads(QUEUE.read_text(encoding="utf-8"))
    if args.backfill_missing:
        touched = 0
        for job in doc["jobs"]:
            if job.get("status") != "done" or args.only not in job.get("id", ""):
                continue
            metrics = job.get("result", {}).get("metrics")
            if not metrics:
                continue
            fresh = parse_all_seeds(job)
            if not fresh:
                continue
            added = [name for name in fresh if name not in metrics]
            if not added:
                continue
            for name in added:
                metrics[name] = fresh[name]
            job["result"]["backfilled_from_log"] = datetime.now(timezone.utc).astimezone().isoformat(
                timespec="seconds"
            )
            touched += 1
            print(f"  {job['id']:52s} 补上 {added}")
        print(f"\n{'(未写回，加 --apply)' if not args.apply else '已写回'}：{touched} 条 job 补了键。")
        if not args.apply or not touched:
            return 0
        QUEUE.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("**请立刻 commit**。")
        return 0

    targets = [
        job
        for job in doc["jobs"]
        if job.get("status") in (None, "pending", "running", "failed")
        and args.only in job.get("id", "")
    ]
    recovered = []
    for job in targets:
        metrics = recoverable(job)
        if metrics is None:
            continue
        recovered.append((job, metrics))
    print(f"候选 {len(targets)} 条，其中可从日志完整恢复 **{len(recovered)}** 条：")
    for job, metrics in recovered:
        score = metrics.get("总得分", {}).get("mean", float("nan"))
        place = metrics.get("名次分", {}).get("mean", float("nan"))
        print(f"  {job['id']:52s} 总得分 {score:+8.3f}  名次分 {place:+7.3f}")
    if not args.apply:
        print("\n（未写回。加 --apply 生效。）")
        return 0
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    for job, metrics in recovered:
        job["status"] = "done"
        job["result"] = {
            "status": "done",
            "seeds": job.get("seeds") or [],
            "matches": job.get("matches"),
            "commands": [
                [
                    "nice", "-n", "15", "uv", "run", "python", "tools/ab_test.py",
                    "--treatment", job["treatment"],
                    "--baseline", job.get("baseline", "heuristic"),
                    "--matches", str(job.get("matches")),
                    "--seed", str(seed),
                    "--jobs", str(job.get("jobs") or 4),
                ]
                + (["--field", job["field"]] if job.get("field") else [])
                for seed in (job.get("seeds") or [])
            ],
            "metrics": metrics,
            "recovered_from_log": stamp,
        }
    QUEUE.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n已写回 {len(recovered)} 条（recovered_from_log={stamp}）。**请立刻 commit**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
