"""廉价可行性核对：三条本线臂（rl / mlp-value / policy-bc）能否在**我的进程内**构造，
并在一个真实局面上 `choose` 出合法动作。不跑对局（省算力），只为确认复核工具可行。

只读主仓 + 读 agent-d 的模型产物；不 import `nnrl.eval`（复核的独立性在测量侧）。
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

MAIN = "/home/wuwenjie01/majiang_ai"
RL = "/home/wuwenjie01/majiang_rl"
sys.path.insert(0, MAIN + "/src")
sys.path.insert(0, RL + "/src")

from majiang.rules.action import legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import round as round_module  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig  # noqa: E402
from majiang.strategy.features import extract  # noqa: E402


def load(rel: str):
    return json.loads((Path(RL) / rel).read_text(encoding="utf-8"))


def build_all():
    out = {}
    t0 = time.time()
    from nnrl.rl_play import RLPolicy

    rl = load("runs/rl-s20260928/params.json")
    out["rl"] = RLPolicy(rl.get("params", rl), sample=False, seed=0, label="rl")
    print(f"  rl         构造 OK ({time.time()-t0:.2f}s)")

    t1 = time.time()
    from nnrl.decider import MLPValueDecider

    mlp = load("runs/mlp-value-s20260928/model.json")
    out["mlp-value"] = MLPValueDecider(
        mlp, HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER)),
        feature_extract=extract, label="mlp-value",
    )
    print(f"  mlp-value  构造 OK ({time.time()-t1:.2f}s)")

    t2 = time.time()
    from nnrl.policy_decider import PolicyNetDecider

    pol = load("runs/policy-bc-v2/model.json")
    out["policy-bc"] = PolicyNetDecider(pol, label="policy-bc")
    print(f"  policy-bc  构造 OK ({time.time()-t2:.2f}s)")
    return out


def main() -> int:
    print("=" * 60)
    print("① 三臂构造")
    arms = build_all()

    print()
    print("② 各臂在真实局面上 choose（判可推理，非对局）")
    rng = random.Random(20260928)
    for name, dec in arms.items():
        acts = []
        while not acts:
            st = round_module.deal(rng, dealer=rng.randrange(4))
            seat = st.turn
            round_module._draw(st, seat)  # noqa: SLF001
            sit = round_module.situation_for(st, seat, PHASE_DRAW)
            acts = legal_actions(sit)
        ch = dec.choose(sit, acts)
        print(f"  {name:10s} choose -> {None if ch is None else ch.describe()}")

    print()
    print("PROBE_ARMS_CONSTRUCT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
