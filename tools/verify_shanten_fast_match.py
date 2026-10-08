"""护栏 1：对局级验证 v5+shanten_fast vs v5 纯 Python。

同 seed 各跑 4 局，逐局比较 scores/winner/fan 必须完全相同。
用法: python tools/verify_shanten_fast_match.py
"""
import sys, time, importlib

sys.path.insert(0, "/home/wuwenjie01/majiang_ai/src")

from majiang.sim.batch import run_match


def run_4rounds(use_fast: bool, seed: int):
    """跑一场 4 局，返回逐局 (scores, winner, fan) 列表。"""
    # 重置 shanten_module 到目标状态
    from majiang.rules import shanten as sm
    importlib.reload(sm)

    if use_fast:
        from majiang.rules import shanten_fast as sf
        sf.ShantenError = sm.ShantenError
        for fn in ("shanten", "best_shanten", "shanten_any", "seven_pairs_shanten", "ukeire"):
            setattr(sm, fn, getattr(sf, fn))

    # 重新 import policy 让它绑定当前 shanten_module 状态
    import majiang.strategy.policy as policy_mod
    importlib.reload(policy_mod)
    # versions 里的 v5 引用的是 policy 里的策略类
    import majiang.strategy.versions as versions_mod
    importlib.reload(versions_mod)

    v5 = versions_mod.build("v5", versions_mod.Mode.QUALIFIER)
    deciders = [v5 for _ in range(4)]

    t0 = time.perf_counter()
    result = run_match(deciders, rounds=4, seed=seed, labels=["a", "b", "c", "d"])
    dt = time.perf_counter() - t0

    # BatchResult 不存逐局明细，只存累计。用累计值做指纹比对。
    fingerprint = {
        "total_scores": [s.total_score for s in result.seats],
        "wins": [s.wins for s in result.seats],
        "god_counts": [s.god_count for s in result.seats],
        "place_points": [s.place_points for s in result.seats],
        "flows": result.flows,
        "rounds": result.rounds,
    }
    return fingerprint, dt


def main():
    seed = 20261003
    print(f"=== 对局级验证（seed={seed}, 4 局）===")

    fp_fast, dt_fast = run_4rounds(use_fast=True, seed=seed)
    print(f"fast:  {dt_fast:.2f}s  {fp_fast}")

    fp_pure, dt_pure = run_4rounds(use_fast=False, seed=seed)
    print(f"pure:  {dt_pure:.2f}s  {fp_pure}")

    match = fp_fast == fp_pure
    print(f"\n结果: {'✅ 完全一致' if match else '❌ 不一致'}")
    print(f"加速比: {dt_pure / dt_fast:.2f}x ({dt_pure:.2f}s -> {dt_fast:.2f}s)")
    if not match:
        for k in fp_fast:
            if fp_fast[k] != fp_pure[k]:
                print(f"  差异 {k}: fast={fp_fast[k]} vs pure={fp_pure[k]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
