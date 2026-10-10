"""报障点**聚合竞技场**：在用户报过的全部局面上，把「用户/豆包主张的那张」与「引擎实选」
放在**同一批确定化世界**里对拍，再跨点聚合。

**为什么要这个工具**（回用户 2026-10-10 14:44 的三条批评）：
1. 「对拍的测试 case 并没有复原当时 4 家的场景」——**成立**：`tools/trigger_counterfactual.py`
   只复原「四家起手 + 真实牌墙 + 到触发点为止的真实事件」，**触发点之后三家由 `--opponents`
   决策器模拟**。本工具把对手换成 **`botlike`**（用真机对局训练的对手模型，77.4% top-1）
   ⇒ 这是当前**最接近真机对手**的可用替代，且对每个点**重采样 N 个世界**（不是沿用真机未来）。
2. 「有可能本身某一家原本就马上要胡牌了，不管我怎么打都不改变结果」——**可以直接量**：
   报告「我方胜率」而不是被锁定局稀释的净分（见 `tools/cf_second_lens.py` 的锁定率口径）。
3. 「如果对拍没有正收益，大概率是改的不对/没理解观点/框架未支持」——本工具的**样本选择
   正是用户自己的点**（带选择偏差，但与用户的关切对齐），与全量对拍互补：**两者结论一致
   才可信；不一致时要说清是哪一侧的选择偏差在起作用**。

**候选的来源**：`--candidates-from-comment` 从报障备注里抓牌码（用户原话里的主张），
并**一定带上引擎实选**作为对照 ⇒ 逐点配对差分 = 「主张 − 引擎」。

用法::
    .venv/bin/python tools/reported_arena.py --samples 200 --jobs 14 --opponents botlike \
        --out agent/out/reports-arena.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import cf_point_mc as MC  # noqa: E402
from majiang.sim import replay as _replay  # noqa: E402

TILE_RE = re.compile(r"[1-9][wtb]|[东南西北中发白]")
SKIP_MARK = ("不处理", "留档", "非报障", "印证")


def _index(code: str) -> int | None:
    """牌码 → 牌种索引（`cf_point_mc._worker` 只接受**索引**，传牌码会 `int('2b')` 炸）。"""
    return _replay._tile_of(code)  # noqa: SLF001


def gather(reports: list[str]) -> list[dict]:
    """从报障快照里取 (报告, 人类座位, 引擎实选, 备注提到的候选)。"""
    out = []
    for path in reports:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        comment = (doc.get("comment") or "").strip()
        if any(mark in comment for mark in SKIP_MARK):
            continue
        decision = doc.get("decision") or {}
        suggestion = (decision.get("suggestion") or {}).get("action") or {}
        picked = suggestion.get("tile")
        if not picked:
            continue
        mentioned = []
        for code in TILE_RE.findall(comment):
            if code not in mentioned:
                mentioned.append(code)
        picked_index = _index(picked)
        if picked_index is None:
            continue
        cands = [picked_index]
        for code in mentioned[:3]:
            value = _index(code)
            if value is not None and value != picked_index and value not in cands:
                cands.append(value)
        out.append({
            "report": path,
            "mine": int((doc.get("session") or {}).get("human_seat", 0)),
            "picked": picked_index,
            "picked_code": picked,
            "mentioned": mentioned[:3],
            "cands": cands[:4],
            "comment": comment[:80],
            "phase": decision.get("phase"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="报障点聚合竞技场（配对 MC）")
    ap.add_argument("--glob", default="webapp/reports/*.json")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--baseline", default="v7")
    ap.add_argument("--opponents", default="botlike")
    ap.add_argument("--mode", default="qualifier")
    ap.add_argument("--out", default="agent/out/reports-arena.json")
    args = ap.parse_args()

    points = gather(sorted(glob.glob(args.glob)))
    print(f"报障点（去掉留档/非报障）{len(points)} 个；对手模型 {args.opponents}；每点 {args.samples} 个世界")
    payloads = [
        (point["report"], point["mine"], point["cands"], args.seed + index, args.baseline,
         args.opponents)
        for point in points
        for index in range(args.samples)
    ]
    with ProcessPoolExecutor(
        max_workers=args.jobs, initializer=MC._init,
        initargs=(args.mode, args.baseline, args.opponents),
    ) as pool:
        results = list(pool.map(MC._worker, payloads, chunksize=4))

    index = 0
    rows = []
    for point in points:
        stats: dict[int, dict[str, list]] = {}
        for _ in range(args.samples):
            for tile, result in results[index].items():
                bucket = stats.setdefault(tile, {"win": [], "score": []})
                if result.get("ok") and result.get("win") is not None:
                    bucket["win"].append(1.0 if result["win"] else 0.0)
                    bucket["score"].append(float(result["score"]))
            index += 1
        entry = {
            "report": Path(point["report"]).name,
            "comment": point["comment"],
            "picked": point["picked"],
            "picked_code": point["picked_code"],
            "per_tile": {},
        }
        for tile, bucket in sorted(stats.items()):
            n = len(bucket["win"])
            if not n:
                continue
            entry["per_tile"][tile] = {
                "n": n,
                "win_rate": sum(bucket["win"]) / n,
                "score": sum(bucket["score"]) / n,
            }
        rows.append(entry)

    # **先落盘再打印**（2026-10-10 17:45 自纠）：上一版在打印时因 `picked_code` 缺失崩掉，
    # 而 `--out` 写在打印之后 ⇒ **50 样本 × 51 点的滚出全白算**。昂贵的部分算完就先存。
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"（明细已落盘 {args.out}）")

    print(f"\n{'报障点':<44}{'引擎':>5}{'候选idx':>8}{'胜率(引擎/候选)':>18}{'期望分(引擎/候选)':>22}")
    diffs_win: list[float] = []
    diffs_score: list[float] = []
    for entry in rows:
        base = entry["per_tile"].get(entry["picked"])
        if not base:
            continue
        for tile, stats in entry["per_tile"].items():
            if tile == entry["picked"]:
                continue
            diffs_win.append(stats["win_rate"] - base["win_rate"])
            diffs_score.append(stats["score"] - base["score"])
            print(f"{entry['report'][:43]:<44}{entry['picked_code']:>5}{tile:>6}"
                  f"{base['win_rate']:>9.1%}/{stats['win_rate']:<8.1%}"
                  f"{base['score']:>11.2f}/{stats['score']:<10.2f}")

    def report(values: list[float], label: str) -> None:
        n = len(values)
        if n < 2:
            print(f"  {label}: 样本不足")
            return
        mean = sum(values) / n
        se = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n)
        print(f"  {label:<14} 均值 {mean:+.4f}  se {se:.4f}  t {mean / se if se else 0:+.2f}"
              f"  MDE(80%) {2.8016 * se:.4f}  n={n}")

    print(f"\n跨点聚合（候选 − 引擎，正值＝用户/豆包的主张更好）：")
    report(diffs_win, "胜率差")
    report(diffs_score, "期望分差")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"逐点明细: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
