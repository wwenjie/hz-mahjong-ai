#!/usr/bin/env python3
"""全量 decider 统计 + 强度排序.

扫描 /home/wuwenjie01/majiang_ai/logs/a_*.jsonl，按 decider 配置串分组：
  - 决策数（decision.made 事件数，字节子串计数，不逐行 JSON 解析）
  - 首次/末次出现时间（runtime.start 的 ts / 文件最后一行的 ts）
  - 覆盖日志份数
  - 局数（= 有 finished 状态 tournament.status 的日志份数）
  - 胜率（最终 score > 0 的比例）
  - 场均/中位得分（最终 tournament.status ranking 中我方 user_id 的 score）

中间结果写 /tmp/decider_intermediate.json；报告写
/home/wuwenjie01/majiang_rl2/outputs/decider_strength_report.md
"""
import glob
import json
import os
import statistics
import sys
import time

LOG_GLOB = "/home/wuwenjie01/majiang_ai/logs/a_*.jsonl"
INTERMEDIATE = "/tmp/decider_intermediate.json"
REPORT = "/home/wuwenjie01/majiang_rl2/outputs/decider_strength_report.md"
LOW_SAMPLE_THRESHOLD = 30  # 局数 < 30 标低置信

B_DECISION = b'"decision.made"'
B_TSTATUS = b'"tournament.status"'
B_FINISHED = b'"status": "finished"'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def scan_file(path):
    """返回 (decider, user_id, ts_first, ts_last, n_decisions, final_score|None).

    final_score: 最后一个 status=finished 的 tournament.status 里，
    ranking 中 user_id 对应条目的第 2 个元素（score）。无则 None。
    """
    decider = None
    user_id = None
    ts_first = None
    ts_last = None
    n_decisions = 0
    final_score = None

    with open(path, "rb") as f:
        first = f.readline()
        if first:
            try:
                e = json.loads(first)
                decider = e.get("decider")
                user_id = e.get("user_id")
                ts_first = e.get("ts")
            except Exception:
                pass

        last_line = None
        for line in f:
            if B_DECISION in line:
                n_decisions += 1
            elif B_TSTATUS in line and B_FINISHED in line:
                try:
                    e = json.loads(line)
                    if e.get("status") == "finished":
                        ranking = e.get("ranking") or []
                        uid = e.get("user_id") or user_id
                        for entry in ranking:
                            if entry and entry[0] == uid:
                                final_score = entry[1]
                                break
                except Exception:
                    pass
            if line.strip():
                last_line = line

        if last_line:
            try:
                ts_last = json.loads(last_line).get("ts")
            except Exception:
                pass

    return decider, user_id, ts_first, ts_last, n_decisions, final_score


def main():
    files = sorted(glob.glob(LOG_GLOB))
    n_files = len(files)
    log(f"发现日志文件 {n_files} 份")

    # per-file 记录（中间数据）
    records = []  # dict: file, decider, ts_first, ts_last, n_decisions, score
    t0 = time.time()
    for i, path in enumerate(files):
        try:
            decider, user_id, ts_first, ts_last, n_dec, score = scan_file(path)
        except Exception as ex:
            log(f"WARN 扫描失败 {os.path.basename(path)}: {ex}")
            continue
        records.append({
            "file": os.path.basename(path),
            "decider": decider,
            "ts_first": ts_first,
            "ts_last": ts_last,
            "n_decisions": n_dec,
            "score": score,
        })
        if (i + 1) % 100 == 0:
            log(f"进度 {i + 1}/{n_files}  耗时 {time.time() - t0:.0f}s")

    log(f"扫描完成，耗时 {time.time() - t0:.0f}s，有效记录 {len(records)}")

    # 中间结果落盘
    with open(INTERMEDIATE, "w") as f:
        json.dump(records, f, ensure_ascii=False)
    log(f"中间结果已写 {INTERMEDIATE} ({os.path.getsize(INTERMEDIATE)} bytes)")

    # 按 decider 聚合
    agg = {}
    for r in records:
        d = r["decider"] or "<missing>"
        a = agg.setdefault(d, {
            "decisions": 0,
            "files": 0,
            "ts_first": None,
            "ts_last": None,
            "scores": [],
            "no_score_files": 0,
        })
        a["decisions"] += r["n_decisions"]
        a["files"] += 1
        if r["ts_first"] and (a["ts_first"] is None or r["ts_first"] < a["ts_first"]):
            a["ts_first"] = r["ts_first"]
        if r["ts_last"] and (a["ts_last"] is None or r["ts_last"] > a["ts_last"]):
            a["ts_last"] = r["ts_last"]
        if r["score"] is not None:
            a["scores"].append(r["score"])
        else:
            a["no_score_files"] += 1

    # 汇总表
    rows = []
    for d, a in agg.items():
        scores = a["scores"]
        games = len(scores)
        wins = sum(1 for s in scores if s > 0)
        win_rate = wins / games if games else None
        avg = statistics.mean(scores) if scores else None
        med = statistics.median(scores) if scores else None
        rows.append({
            "decider": d,
            "decisions": a["decisions"],
            "files": a["files"],
            "games": games,
            "no_score_files": a["no_score_files"],
            "win_rate": win_rate,
            "avg_score": avg,
            "median_score": med,
            "ts_first": a["ts_first"],
            "ts_last": a["ts_last"],
        })
    # 按局数降序，再按场均得分降序
    rows.sort(key=lambda r: (-r["games"], -(r["avg_score"] or -1e9)))

    # 写报告
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Decider 强度报告（全量日志扫描）\n\n")
        f.write(f"- 日志总数（`logs/a_*.jsonl`）：**{n_files}** 份\n")
        f.write(f"- 有效扫描：{len(records)} 份\n")
        f.write(f"- 无最终得分的日志：{sum(1 for r in records if r['score'] is None)} 份\n")
        f.write(f"- 得分口径：每份日志最后一条 `status=finished` 的 `tournament.status` "
                f"事件中 ranking 里我方 user_id 的 score（整场 10 局 × 8 轮的总分）\n")
        f.write(f"- 低置信阈值：局数 < {LOW_SAMPLE_THRESHOLD}\n\n")
        f.write("| decider 配置串 | 决策数 | 日志份数 | 有效场数 | 胜率 | 场均得分 | 中位得分 | 首次出现 | 末次出现 | 建议 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r["decider"] == "first-legal":
                advice = "剔除"
            elif r["games"] < LOW_SAMPLE_THRESHOLD:
                advice = "低置信"
            else:
                advice = "候选"  # 拍板由报告读者做
            wr = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "—"
            avg = f"{r['avg_score']:.1f}" if r["avg_score"] is not None else "—"
            med = f"{r['median_score']:.0f}" if r["median_score"] is not None else "—"
            dec = r["decider"].replace("|", "\\|")
            f.write(f"| `{dec}` | {r['decisions']} | {r['files']} | {r['games']} "
                    f"| {wr} | {avg} | {med} | {r['ts_first'] or '—'} "
                    f"| {r['ts_last'] or '—'} | {advice} |\n")
    log(f"报告已写 {REPORT}")

    # stdout 输出汇总表供日志查看
    print("\n=== 汇总 ===")
    for r in rows:
        wr = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "—"
        avg = f"{r['avg_score']:.1f}" if r["avg_score"] is not None else "—"
        print(f"games={r['games']:4d} files={r['files']:4d} win={wr:>6} "
              f"avg={avg:>8} dec={r['decisions']:7d}  {r['decider']}")


if __name__ == "__main__":
    main()
