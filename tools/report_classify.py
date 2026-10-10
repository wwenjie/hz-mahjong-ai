"""报障点**逐点定因器**：把用户报的每个点按「可判定的机制」分到互斥的类别里，并回答
「哪一类改动能救它」。

**为什么需要它**（用户 2026-10-10 15:07）：「我的报障包含了不同的场景，有些是同原因的，
有些原因不同，但都是很明显的决策问题」——**对**。只说「全落在形质/排序与吃碰闸门两条线」
是把不同机制糊成两桶，既不能追查也不能验收。本工具按**决策链路**逐段问：

| 类别 | 判定（全部可计算） | 该由哪条轴管 |
|---|---|---|
| **C6 副露窗口** | 阶段不是出牌（吃/碰/杠窗口） | 吃碰闸门（`meld_tolerance`） |
| **C7 胡/弃胡** | 阶段是胡牌判定 | 弃胡阈值（`piao_threshold_scale`） |
| **C1 破平层覆盖主分** | 实选 ≠ `argmax total` | 破平层收口（已测：−0.413 ⇒ 保留） |
| **C5 候选面截断** | 用户那张 **∈ 全层最大进张** 但 **∉ 形质前 3** | **候选面放宽（`ukeire_candidates` → cand5）** |
| **C2 形质并列** | 实选 = `argmax total` 且该层形质全并列 | 破平层/精确进张（已测：无剩余空间） |
| **C3 判据相同（真平局）** | 用户那张与实选的 `(total, 进张)` **完全相同** | **无判据**：只能靠新判据（进张后牌形等） |
| **C4 主分严格最优** | 实选 = `argmax total` 且严格优于用户那张 | 口味/估值（`total` 的构造） |

另附「**哪个臂会打出用户那张**」的计数（`v7-cand5` / `v7-tiedfull` / `v5-maxtotal` / `v7m`），
这是「这条轴能不能救这个点」的直接读数。

用法::
    .venv/bin/python tools/report_classify.py --csv agent/out/report-classify.csv
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import shanten as sh  # noqa: E402
from majiang.rules import tiles as T  # noqa: E402
from majiang.rules.action import CHI, DISCARD, GANG, PENG, legal_actions  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402
from replay_report import _situation_from_view  # noqa: E402

TILE_RE = re.compile(r"[1-9][wtb]|[东南西北中发白]")
SKIP_MARK = ("不处理", "留档", "非报障", "印证")
ARMS = ("v7", "v7-cand5", "v7-tiedfull", "v5-maxtotal", "v7m", "v7-pairs")


def claimed(comment: str) -> list[str]:
    """抽「用户主张」的牌码。

    **必须避开用户粘贴的豆包分析正文**（那里面满是牌码，会把「主张」淹掉）。
    口径：只取备注**前 80 字**里、且**紧跟在「打/弃/舍/留/优先」之后**的牌码；
    拿不到就返回空（宁可空，也不要拿错——空的点不参与 C3/C5 的判定）。
    """
    head = (comment or "")[:80]
    out = []
    for match in re.finditer(r"(?:打|弃|舍|出|优先打?)[^\u4e00-\u9fff]{0,3}([1-9][wtb]|[东南西北中发白])", head):
        code = match.group(1)
        if code not in out:
            out.append(code)
    if not out:
        for code in TILE_RE.findall(head):
            if code not in out:
                out.append(code)
    return out


# 手工校正：备注开头就是 AI 正文、或主张是「碰/吃」的点（A 2026-10-10 15:20 逐条看过）。
CLAIM_OVERRIDE = {
    "report_20261008_235410_seq22.json": "4w",   # 用户：4w 是完全孤立单张
    "report_20261009_154828_seq22.json": "6b",   # 用户：优先打掉绝对单牌 6b
    "report_20261010_120431_seq21.json": "7b",
    "report_20261010_120616_seq23.json": "3b",
    "report_20261010_120735_seq24.json": "9t",   # 碰 9t
    "report_20261009_001617_seq53.json": "发",
    "report_20261009_001619_seq53.json": "发",
    "report_20261009_001758_seq55.json": "2t",
    "report_20261009_001759_seq55.json": "2t",
    "report_20261009_001922_seq57.json": "6t",
    "report_20261009_001923_seq57.json": "6t",
    "report_20261009_114042_seq135.json": "8w",
    "report_20261009_114043_seq135.json": "8w",
    "report_20261009_133127_seq163.json": "9t",
    "report_20261009_133128_seq163.json": "9t",
    "report_20261009_133235_seq164.json": "8t",
    "report_20261009_143128_seq197.json": "1b",
    "report_20261009_102559_seq71.json": "7w",
    "report_20261009_111442_seq113.json": "2t",
    "report_20261009_111444_seq113.json": "2t",
    "report_20261009_150000_seq17.json": "1b",
    "report_20261009_154048_seq17.json": "1b",
    "report_20261009_114836_seq141.json": "3t",
    "report_20261009_132408_seq160.json": "6t",
    "report_20261009_132409_seq160.json": "6t",
    "report_20261009_112221_seq124.json": "1b",
    "report_20261009_112223_seq124.json": "1b",
    "report_20261009_111956_seq121.json": "9t",
    "report_20261009_111957_seq121.json": "9t",
    "report_20261009_143856_seq198.json": "6t",
    "report_20261009_144114_seq201.json": "7t",
    "report_20261009_144115_seq201.json": "7t",
    "report_20261009_103815_seq78.json": "9t",
    "report_20261009_105200_seq80.json": "2t",
    "report_20261009_002245_seq62.json": "吃",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="报障点逐点定因")
    ap.add_argument("--glob", default="webapp/reports/*.json")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    deciders = {arm: make_decider(arm, Mode.QUALIFIER) for arm in ARMS}
    rows = []
    stats: collections.Counter = collections.Counter()
    for path in sorted(glob.glob(args.glob)):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        comment = (doc.get("comment") or "").strip()
        if any(mark in comment for mark in SKIP_MARK):
            continue
        decision = doc.get("decision") or {}
        view = decision.get("situation") or {}
        try:
            sit = _situation_from_view(view)
        except Exception as error:  # noqa: BLE001
            stats["不可重建"] += 1
            continue
        actions = tuple(legal_actions(sit))
        picks = {}
        for arm, decider in deciders.items():
            try:
                choice = decider.choose(sit, actions, budget_ms=2000)
                picks[arm] = choice.describe() if choice else ""
            except Exception:  # noqa: BLE001
                picks[arm] = ""
        mine = picks.get("v7", "")
        override = CLAIM_OVERRIDE.get(Path(path).name)
        want = [override] if override else claimed(comment)
        row = {
            "report": Path(path).name,
            "ts": doc.get("created_at", "")[5:16],
            "phase": sit.phase,
            "hand": " ".join(view.get("my_hand") or []),
            "v7": mine,
            "claimed": "/".join(want[:3]),
            "comment": comment[:60],
        }
        row.update({arm: picks.get(arm, "") for arm in ARMS})

        # ── 定因 ────────────────────────────────────────────────────────
        if sit.phase != "draw":
            kinds = {a.kind for a in actions}
            row["kind"] = "C7 胡/弃胡" if "hu" in kinds or "piao" in kinds else "C6 副露窗口"
        else:
            discards = [a for a in actions if a.kind == DISCARD]
            scores = [deciders["v7"]._score_discard(sit, a) for a in discards]  # noqa: SLF001
            top = min(item.shanten for item in scores)
            layer = [item for item in scores if item.shanten == top]
            argmax_total = max(layer, key=lambda item: item.total)
            # **必须按「分值」判 C1，不能按「牌」判**：`max()` 在并列时按顺序取，
            # 拿 `.tile` 去比会把每一个「主分并列」的点都误判成「破平层覆盖主分」
            # （2026-10-10 15:15 实测：误判出 17 个 C1，真实只有 4 个）。
            max_total = argmax_total.total
            tie_tol = 1e-6
            # 形质（blocks）排序后的前 `ukeire_candidates` 张 = 真正进入精确进张比较的面
            by_blocks = sorted(layer, key=lambda item: -item.blocks)[:3]
            in_face = {item.tile for item in by_blocks}
            hand = list(sit.hand.counts)
            ukeire = {}
            for item in layer:
                counts = list(hand)
                counts[item.tile] -= 1
                try:
                    got = sh.ukeire(counts, sit.hand.meld_count)
                    ukeire[item.tile] = sum(got[1]) if isinstance(got, tuple) and len(got) > 1 \
                        and isinstance(got[1], (list, tuple)) else 0
                except Exception:  # noqa: BLE001
                    ukeire[item.tile] = 0
            best_ukeire = max(ukeire.values()) if ukeire else 0
            picked_tile = None
            if mine.startswith("discard:"):
                code = mine.split(":", 1)[1]
                for a in discards:
                    if T.to_code(a.tile) == code:
                        picked_tile = a.tile
            want_tiles = []
            for code in want:
                for a in discards:
                    if T.to_code(a.tile) == code:
                        want_tiles.append(a.tile)
            picked_score = next((i for i in layer if i.tile == picked_tile), None)
            row["gap"] = round(max_total - (picked_score.total if picked_score else max_total), 2)
            if picked_tile is None:
                row["kind"] = "其他"
            elif picked_score is not None and picked_score.total < max_total - tie_tol:
                row["kind"] = "C1 破平层覆盖主分"
            elif any(
                want_tile not in in_face and ukeire.get(want_tile, -1) == best_ukeire
                for want_tile in want_tiles
            ):
                row["kind"] = "C5 候选面截断(cand5可救)"
            elif len({i.blocks for i in layer}) == 1:
                row["kind"] = "C2 形质并列"
            elif want_tiles and all(
                ukeire.get(w, -1) == ukeire.get(picked_tile, -2)
                and abs(next((i.total for i in layer if i.tile == w), 0)
                        - (picked_score.total if picked_score else 0)) < tie_tol
                for w in want_tiles
            ):
                row["kind"] = "C3 判据相同(真平局)"
            else:
                row["kind"] = "C4 主分严格最优"
        stats[row["kind"]] += 1
        rows.append(row)

    print(f"报障点 {len(rows)} 个（已去掉留档/非报障）")
    print("\n类别分布：")
    for kind, count in stats.most_common():
        print(f"  {kind:<26} {count:>3}（{count / max(1, len(rows)):.1%}）")
    print("\n**哪个臂会打出用户主张的那张**（逐点命中数）：")
    for arm in ARMS:
        hit = sum(1 for row in rows if row["claimed"] and any(
            row[arm].endswith(":" + code) for code in row["claimed"].split("/") if code))
        print(f"  {arm:<16}{hit:>3} / {len(rows)}")
    print("\n逐点明细：")
    for row in rows:
        print(f"  {row['ts']} {row['report'][:34]:<36}{row['kind']:<24}"
              f"v7={row['v7']:<12} 你要={row['claimed']:<10} "
              f"cand5={row['v7-cand5']:<12}")
    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
