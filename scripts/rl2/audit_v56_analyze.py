#!/usr/bin/env python
"""阶段二审计：一致率分析 + 系统性错误模式识别。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]
TILE_NAMES = [
    "1万", "2万", "3万", "4万", "5万", "6万", "7万", "8万", "9万",
    "1筒", "2筒", "3筒", "4筒", "5筒", "6筒", "7筒", "8筒", "9筒",
    "1条", "2条", "3条", "4条", "5条", "6条", "7条", "8条", "9条",
    "东", "南", "西", "北", "中", "发", "白",
]


def fmt_tile(t: int) -> str:
    if 0 <= t < 34:
        return TILE_NAMES[t]
    return f"?{t}"


def hand_str(hand_counts: np.ndarray) -> str:
    parts = []
    for t in range(34):
        c = int(hand_counts[t])
        if c > 0:
            parts.append(f"{fmt_tile(t)}×{c}" if c > 1 else fmt_tile(t))
    return " ".join(parts)


def load_data():
    d = np.load("data/audit_v56_with_model.npz", allow_pickle=True)
    return {k: d[k] for k in d.keys()}


def shanten_bucket(s: np.ndarray) -> np.ndarray:
    """-1->0(听牌), 0->1(和了), 1->2, 2->3, 3+->4"""
    b = np.full(len(s), 4, dtype=np.int8)
    b[s == -1] = 0
    b[s == 0] = 1
    b[s == 1] = 2
    b[s == 2] = 3
    b[s >= 3] = 4
    return b


def round_bucket(r: np.ndarray) -> np.ndarray:
    """1-3 早期, 4-6 中期, 7-8 晚期"""
    b = np.full(len(r), 2, dtype=np.int8)
    b[r <= 3] = 0
    b[(r >= 4) & (r <= 6)] = 1
    return b


def score_bucket(s: np.ndarray) -> np.ndarray:
    """输(<-20), 平(-20..20), 赢(>20)"""
    b = np.full(len(s), 1, dtype=np.int8)
    b[s < -20] = 0
    b[s > 20] = 2
    return b


def consistency_report(d):
    n = len(d["chosen_action_kind"])
    chosen_a = d["chosen_action_kind"]
    top1_a = d["model_top1_action"]
    top1_a_raw = d["model_top1_action_raw"]
    prob_chosen = d["model_prob_chosen"]
    prob_chosen_masked = d["model_prob_chosen_masked"]
    score = d["game_final_score"]

    match = top1_a == chosen_a
    match_raw = top1_a_raw == chosen_a

    lines = []
    lines.append("## 总体一致率\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 样本数 | {n:,} |")
    lines.append(f"| top1 一致率（掩码后） | {match.mean():.4f} |")
    lines.append(f"| top1 一致率（原始） | {match_raw.mean():.4f} |")
    lines.append(f"| 模型置信度（原始概率均值） | {prob_chosen.mean():.4f} |")
    lines.append(f"| 模型置信度（掩码归一化均值） | {prob_chosen_masked.mean():.4f} |")
    lines.append("")

    # By phase
    lines.append("### 按 phase\n")
    lines.append("| phase | 样本数 | 一致率 | 置信度(掩码) | 平均得分 |")
    lines.append("|---|---|---|---|---|")
    for ph in ["draw", "response_peng", "response_chi"]:
        mask = d["phase"] == ph
        if mask.sum() == 0:
            continue
        lines.append(
            f"| {ph} | {mask.sum():,} | {match[mask].mean():.4f} | "
            f"{prob_chosen_masked[mask].mean():.4f} | {score[mask].mean():.1f} |"
        )
    lines.append("")

    # By shanten
    lines.append("### 按 shanten\n")
    lines.append("| shanten | 样本数 | 一致率 | 置信度(掩码) | 平均得分 |")
    lines.append("|---|---|---|---|---|")
    sb = shanten_bucket(d["shanten"])
    sb_names = ["听牌(-1)", "和了(0)", "1向听", "2向听", "3+向听"]
    for i, name in enumerate(sb_names):
        mask = sb == i
        if mask.sum() == 0:
            continue
        lines.append(
            f"| {name} | {mask.sum():,} | {match[mask].mean():.4f} | "
            f"{prob_chosen_masked[mask].mean():.4f} | {score[mask].mean():.1f} |"
        )
    lines.append("")

    # By chosen_action_kind
    lines.append("### 按 chosen_action_kind\n")
    lines.append("| 动作 | 样本数 | 一致率 | 置信度(掩码) | 平均得分 |")
    lines.append("|---|---|---|---|---|")
    for a_id, a_name in enumerate(ACTION_NAMES):
        mask = chosen_a == a_id
        if mask.sum() == 0:
            continue
        lines.append(
            f"| {a_name} | {mask.sum():,} | {match[mask].mean():.4f} | "
            f"{prob_chosen_masked[mask].mean():.4f} | {score[mask].mean():.1f} |"
        )
    lines.append("")

    # By round_no
    lines.append("### 按 round_no\n")
    lines.append("| 阶段 | 样本数 | 一致率 | 置信度(掩码) | 平均得分 |")
    lines.append("|---|---|---|---|---|")
    rb = round_bucket(d["round_no"])
    rb_names = ["早期(1-3)", "中期(4-6)", "晚期(7-8)"]
    for i, name in enumerate(rb_names):
        mask = rb == i
        if mask.sum() == 0:
            continue
        lines.append(
            f"| {name} | {mask.sum():,} | {match[mask].mean():.4f} | "
            f"{prob_chosen_masked[mask].mean():.4f} | {score[mask].mean():.1f} |"
        )
    lines.append("")

    # By score bucket
    lines.append("### 按得分分组\n")
    lines.append("| 得分组 | 样本数 | 一致率 | 置信度(掩码) |")
    lines.append("|---|---|---|---|")
    sc = score_bucket(score)
    sc_names = ["输局(<-20)", "平局(-20~20)", "赢局(>20)"]
    for i, name in enumerate(sc_names):
        mask = sc == i
        if mask.sum() == 0:
            continue
        lines.append(
            f"| {name} | {mask.sum():,} | {match[mask].mean():.4f} | "
            f"{prob_chosen_masked[mask].mean():.4f} |"
        )
    lines.append("")

    return lines


def error_patterns(d):
    """找出系统性错误模式。"""
    n = len(d["chosen_action_kind"])
    chosen_a = d["chosen_action_kind"]
    prob_chosen_masked = d["model_prob_chosen_masked"]
    score = d["game_final_score"]
    shanten = d["shanten"]
    round_no = d["round_no"]
    phase = d["phase"]
    god = d["obs_god"]
    hand = d["obs_hand"]

    # 财神数（手牌中 god tile 的数量）
    god_tile = god  # all 33 (白板)
    god_count = hand[np.arange(n), god_tile].astype(np.int32)

    # 选错定义：模型给实际选择的掩码归一化概率 < 10% 且最终得分 < 0
    wrong = (prob_chosen_masked < 0.10) & (score < 0)
    correct = (prob_chosen_masked >= 0.10) & (score >= 0)

    lines = []
    lines.append("## 系统性错误模式\n")
    lines.append(f"- 选错样本数（模型不认同且输分）: {wrong.sum():,} / {n:,} ({wrong.mean():.4f})")
    lines.append(f"- 选对样本数（模型认同且赢/平分）: {correct.sum():,} / {n:,} ({correct.mean():.4f})")
    lines.append("")

    # 按维度分组统计选错率
    dims = []
    dims.append(("phase", phase, None))
    dims.append(("shanten", shanten, None))
    dims.append(("round_no", round_no, None))
    dims.append(("chosen_action", chosen_a, ACTION_NAMES))
    dims.append(("god_count", god_count, None))

    # 组合维度：phase × shanten
    lines.append("### 按 phase × shanten 选错率\n")
    lines.append("| phase | shanten | 样本数 | 选错率 | 平均得分 |")
    lines.append("|---|---|---|---|---|")
    for ph in ["draw", "response_peng", "response_chi"]:
        ph_mask = phase == ph
        for sv in sorted(np.unique(shanten)):
            mask = ph_mask & (shanten == sv)
            if mask.sum() < 100:
                continue
            wr = wrong[mask].mean()
            lines.append(
                f"| {ph} | {sv} | {mask.sum():,} | {wr:.4f} | {score[mask].mean():.1f} |"
            )
    lines.append("")

    # 组合维度：phase × action
    lines.append("### 按 phase × chosen_action 选错率\n")
    lines.append("| phase | action | 样本数 | 选错率 | 平均得分 |")
    lines.append("|---|---|---|---|---|")
    for ph in ["draw", "response_peng", "response_chi"]:
        ph_mask = phase == ph
        for a_id, a_name in enumerate(ACTION_NAMES):
            mask = ph_mask & (chosen_a == a_id)
            if mask.sum() < 100:
                continue
            wr = wrong[mask].mean()
            lines.append(
                f"| {ph} | {a_name} | {mask.sum():,} | {wr:.4f} | {score[mask].mean():.1f} |"
            )
    lines.append("")

    # Top 10 错误模式（按选错率 × 样本数加权）
    lines.append("### Top 10 系统性错误模式\n")
    lines.append("| # | 模式 | 样本数 | 选错率 | 平均得分 | 置信度(掩码) |")
    lines.append("|---|---|---|---|---|---|")

    patterns = []
    # phase × shanten × action
    for ph in ["draw", "response_peng", "response_chi"]:
        ph_mask = phase == ph
        for sv in sorted(np.unique(shanten)):
            for a_id in range(6):
                mask = ph_mask & (shanten == sv) & (chosen_a == a_id)
                if mask.sum() < 200:
                    continue
                wr = wrong[mask].mean()
                if wr < 0.01:
                    continue
                patterns.append({
                    "desc": f"{ph} shanten={sv} action={ACTION_NAMES[a_id]}",
                    "count": mask.sum(),
                    "wrong_rate": wr,
                    "avg_score": score[mask].mean(),
                    "avg_conf": prob_chosen_masked[mask].mean(),
                    "mask": mask,
                })

    # phase × round × action
    for ph in ["draw", "response_peng", "response_chi"]:
        ph_mask = phase == ph
        for rv in sorted(np.unique(round_no)):
            for a_id in range(6):
                mask = ph_mask & (round_no == rv) & (chosen_a == a_id)
                if mask.sum() < 200:
                    continue
                wr = wrong[mask].mean()
                if wr < 0.01:
                    continue
                patterns.append({
                    "desc": f"{ph} round={rv} action={ACTION_NAMES[a_id]}",
                    "count": mask.sum(),
                    "wrong_rate": wr,
                    "avg_score": score[mask].mean(),
                    "avg_conf": prob_chosen_masked[mask].mean(),
                    "mask": mask,
                })

    # phase × god_count × action
    for ph in ["draw", "response_peng", "response_chi"]:
        ph_mask = phase == ph
        for gc in sorted(np.unique(god_count)):
            for a_id in range(6):
                mask = ph_mask & (god_count == gc) & (chosen_a == a_id)
                if mask.sum() < 200:
                    continue
                wr = wrong[mask].mean()
                if wr < 0.01:
                    continue
                patterns.append({
                    "desc": f"{ph} god_count={gc} action={ACTION_NAMES[a_id]}",
                    "count": mask.sum(),
                    "wrong_rate": wr,
                    "avg_score": score[mask].mean(),
                    "avg_conf": prob_chosen_masked[mask].mean(),
                    "mask": mask,
                })

    # Sort by wrong_rate * sqrt(count) for balance
    patterns.sort(key=lambda x: x["wrong_rate"] * np.sqrt(x["count"]), reverse=True)

    for i, p in enumerate(patterns[:10]):
        lines.append(
            f"| {i+1} | {p['desc']} | {p['count']:,} | {p['wrong_rate']:.4f} | "
            f"{p['avg_score']:.1f} | {p['avg_conf']:.4f} |"
        )
    lines.append("")

    return lines, patterns[:10]


def case_studies(d, patterns):
    """对 top 错误模式打印具体案例。"""
    lines = []
    lines.append("## 具体案例\n")

    chosen_a = d["chosen_action_kind"]
    chosen_t = d["chosen_tile"]
    top1_a = d["model_top1_action"]
    top1_t = d["model_top1_tile"]
    prob_chosen_masked = d["model_prob_chosen_masked"]
    score = d["game_final_score"]
    hand = d["obs_hand"]
    shanten = d["shanten"]
    round_no = d["round_no"]
    phase = d["phase"]
    target = d["obs_target"]
    seat = d["obs_seat"]

    for pi, p in enumerate(patterns[:5]):
        lines.append(f"### 模式 {pi+1}: {p['desc']}\n")
        mask = p["mask"]
        wrong_mask = mask & (prob_chosen_masked < 0.10) & (score < 0)
        idxs = np.where(wrong_mask)[0][:3]
        for i, idx in enumerate(idxs):
            h = hand[idx]
            hand_desc = hand_str(h)
            lines.append(f"**案例 {i+1}** (idx={idx}):")
            lines.append(f"- phase={phase[idx]}, shanten={shanten[idx]}, round={round_no[idx]}, seat={seat[idx]}, target={fmt_tile(int(target[idx]))}")
            lines.append(f"- 手牌: {hand_desc}")
            lines.append(f"- 规则引擎选择: {ACTION_NAMES[chosen_a[idx]]}" + (f" tile={fmt_tile(int(chosen_t[idx]))}" if chosen_a[idx] == 0 else ""))
            lines.append(f"- 模型 top1: {ACTION_NAMES[top1_a[idx]]}" + (f" tile={fmt_tile(int(top1_t[idx]))}" if top1_a[idx] == 0 else ""))
            lines.append(f"- 模型给规则引擎选择的概率(掩码): {prob_chosen_masked[idx]:.4f}")
            lines.append(f"- 该局最终得分: {score[idx]}")
            lines.append("")

    return lines


def counterfactual(d, patterns):
    """反事实分析：比较选模型 top1 vs 选规则引擎的局的平均得分。"""
    lines = []
    lines.append("## 反事实分析\n")
    lines.append("> 注：同一 game 无法重来。以下比较的是「模型 top1 与规则引擎一致的局」vs「不一致的局」的平均得分差异，作为近似反事实。\n")

    chosen_a = d["chosen_action_kind"]
    top1_a = d["model_top1_action"]
    score = d["game_final_score"]
    prob_chosen_masked = d["model_prob_chosen_masked"]

    agree = top1_a == chosen_a
    disagree = ~agree

    lines.append("| 分组 | 样本数 | 平均得分 | 中位得分 | 胜率(>0) |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 模型与规则一致 | {agree.sum():,} | {score[agree].mean():.1f} | {np.median(score[agree]):.0f} | {(score[agree] > 0).mean():.4f} |")
    lines.append(f"| 模型与规则不一致 | {disagree.sum():,} | {score[disagree].mean():.1f} | {np.median(score[disagree]):.0f} | {(score[disagree] > 0).mean():.4f} |")
    lines.append("")

    # For top 3 patterns, compare agree vs disagree within each
    lines.append("### Top 3 错误模式内的反事实\n")
    lines.append("| 模式 | 一致局数 | 一致局均分 | 不一致局数 | 不一致局均分 | 差值 |")
    lines.append("|---|---|---|---|---|---|")
    for pi, p in enumerate(patterns[:3]):
        mask = p["mask"]
        agree_m = mask & agree
        disagree_m = mask & disagree
        if agree_m.sum() == 0 or disagree_m.sum() == 0:
            continue
        diff = score[disagree_m].mean() - score[agree_m].mean()
        lines.append(
            f"| {p['desc']} | {agree_m.sum():,} | {score[agree_m].mean():.1f} | "
            f"{disagree_m.sum():,} | {score[disagree_m].mean():.1f} | {diff:+.1f} |"
        )
    lines.append("")

    return lines


def recommendations(d, patterns):
    lines = []
    lines.append("## 对启发式改进的建议\n")

    chosen_a = d["chosen_action_kind"]
    prob_chosen_masked = d["model_prob_chosen_masked"]
    score = d["game_final_score"]

    # 1. 按动作类型看选错率
    wrong = (prob_chosen_masked < 0.10) & (score < 0)
    for a_id, a_name in enumerate(ACTION_NAMES):
        mask = chosen_a == a_id
        if mask.sum() == 0:
            continue
        wr = wrong[mask].mean()
        if wr > 0.05:
            lines.append(f"- **{a_name} 动作选错率 {wr:.4f}**（{mask.sum():,} 样本）：规则引擎的 {a_name} 决策被模型强烈不认同，建议重新审视该动作的规则逻辑。")

    # 2. 按 phase
    for ph in ["draw", "response_peng", "response_chi"]:
        mask = d["phase"] == ph
        wr = wrong[mask].mean()
        if wr > 0.05:
            lines.append(f"- **{ph} 阶段选错率 {wr:.4f}**：该阶段决策逻辑可能是主要短板。")

    # 3. 按 shanten
    sb = shanten_bucket(d["shanten"])
    sb_names = ["听牌(-1)", "和了(0)", "1向听", "2向听", "3+向听"]
    for i, name in enumerate(sb_names):
        mask = sb == i
        wr = wrong[mask].mean()
        if wr > 0.05:
            lines.append(f"- **{name} 选错率 {wr:.4f}**：该向听数下的决策需要改进。")

    # 4. 高分歧动作
    lines.append("")
    lines.append("### 模型最反对规则引擎的动作（按置信度低 + 样本多排序）\n")
    lines.append("| 动作 | 样本数 | 平均置信度(掩码) | 选错率 | 平均得分 |")
    lines.append("|---|---|---|---|---|")
    for a_id, a_name in enumerate(ACTION_NAMES):
        mask = chosen_a == a_id
        if mask.sum() == 0:
            continue
        lines.append(
            f"| {a_name} | {mask.sum():,} | {prob_chosen_masked[mask].mean():.4f} | "
            f"{wrong[mask].mean():.4f} | {score[mask].mean():.1f} |"
        )
    lines.append("")

    # 5. 通用建议
    lines.append("### 通用建议\n")
    lines.append("1. **discard 决策占 57%**：如果 discard 选错率高，优先改进出牌策略（如引入安全牌判断、对手立直检测）。")
    lines.append("2. **pass 决策占 38%**：pass 过多说明规则引擎可能过于保守，或模型在响应阶段更激进。")
    lines.append("3. **低 shanten 局面**：如果听牌/1向听选错率高，说明规则引擎在接近和牌时的决策不够精准。")
    lines.append("4. **晚期局面**：如果晚期选错率高，说明规则引擎的残局处理（如防守、弃和）需要加强。")
    lines.append("")

    return lines


def main():
    print("Loading data...")
    d = load_data()
    print(f"Loaded {len(d['chosen_action_kind']):,} samples")

    report_lines = []
    report_lines.append("# 启发式系统性错误审计报告\n")
    report_lines.append("> 数据: audit_v56.npz (1,053,763 样本) | 模型: bc_v7_base.pt (19.2% 胜率)\n")

    print("Computing consistency...")
    report_lines.extend(consistency_report(d))

    print("Identifying error patterns...")
    pattern_lines, top_patterns = error_patterns(d)
    report_lines.extend(pattern_lines)

    print("Generating case studies...")
    report_lines.extend(case_studies(d, top_patterns))

    print("Running counterfactual analysis...")
    report_lines.extend(counterfactual(d, top_patterns))

    print("Generating recommendations...")
    report_lines.extend(recommendations(d, top_patterns))

    out_path = Path("outputs/audit_v56_report.md")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
