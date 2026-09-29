"""牌墙、可摸/可杠与流局判定。

牌墙构成：全场 136 张，发牌消耗 53 张（四家各 13 张 + 庄家第 14 张直抽），
余下 83 张构成牌墙；其中最后 10 墩（20 张）保留不摸，故本局最多可摸 63 张。

「可摸」与「可杠」的判据相同：两者都需要从牌墙取走一张，因此当牌墙只剩保留的
20 张时，既不能摸牌也不能杠牌——这正是「最后 10 墩之内禁止杠牌」的由来。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import tiles

SEATS = 4
DEALT_TILES = tiles.HAND_SIZE * SEATS + 1
INITIAL_WALL = tiles.WALL_SIZE - DEALT_TILES
# 一场几局。**由服务端 room_config 的 `Rounds` 给出**（实测恒为 8），这里只是缺省值；
# 真机路径应注入真实值，否则「剩余局数」会算错。
ROUNDS_PER_GAME_DEFAULT = 8
RESERVED_TILES = 20
MAX_DRAWS = INITIAL_WALL - RESERVED_TILES


class TableError(ValueError):
    """对局状态非法。"""


@dataclass(frozen=True, slots=True)
class TableState:
    wall_remaining: int = INITIAL_WALL
    dealer_seat: int = 0
    round_no: int = 1
    # **局况**：四家累计比分（座位序）与本场总局数。
    #
    # 为什么要有这两个字段：比赛按**首名率**结算（我们 6.2% vs 榜首 45.0%），而决策器此前
    # **完全看不到比分**——`TableState` 只有牌墙/庄家/局号。实测（`tools/analyze_match_standing.py`，
    # 4800 局）：`P(终局首名 | 本局开始时的名次 × 剩余局数)` 从 **2.3%**（垫底+剩≤3 局）
    # 到 **69.9%**（领先+剩≤3 局）——**30 倍跨度**，而这个变量一直没进决策器。
    # 两个极端档位的战略含义相反（领先且近尾该保、垫底且近尾该搏），而此前两处打得一模一样。
    #
    # 平台**本来就发比分**（`Snapshot.scores`，`client/snapshot.py:187`，也在实测字段表 `:210` 里），
    # 只是 `to_situation()` 把它丢了。所以这是**补上一个已收到的输入**，不是新造数据。
    #
    # 缺省空元组 = 「局况未知」（例如单局离线调用），消费方必须**当作未知处理**，
    # 不得把空元组当成「四家都是 0 分」——那会让所有离线旧路径突然以为自己在并列第一。
    scores: tuple[int, ...] = ()
    rounds_total: int = ROUNDS_PER_GAME_DEFAULT
    # **自对弈侧还没接上（已知缺口，2026-09-30 02:20）**：真机路径已通并测试覆盖
    # （`client/snapshot.py:table_state` → `tests/test_snapshot.py`），但自对弈路径没走通——
    # 试过在 `sim/round.py:run_round` 里把 `prior_scores` 写进 `RoundState.scores`、
    # `sim/batch.py:run_match` 传累计分，实测探针拿到的仍是全 0，且**不是**「没人胡」造成的
    # （8 局 0 流局、四家总分 −23/−13/34/2，但 344 条记录里累计分全为 0）。
    # 线索：`deal()` 之后 `state.table` 会因为牌墙 84 抛 `TableError`（尚未摸第一张牌），
    # 所以 `play_round` 构造局面的那条路径与 `RoundState.table` **不是同一条**——
    # 下一步应先定位 `play_round` 实际用哪条路径造 `Situation`。
    # **影响**：在接上之前，任何「局况类档位」在自对弈里看到的都是空元组 ⇒ 必然测成空干预。
    # 所以**先别开局况臂**，或者先把这条接线补完。

    def __post_init__(self) -> None:
        if not 0 <= self.wall_remaining <= INITIAL_WALL:
            raise TableError(f"牌墙剩余张数越界: {self.wall_remaining}")
        if not 0 <= self.dealer_seat < SEATS:
            raise TableError(f"庄家座位越界: {self.dealer_seat}")
        if self.round_no < 1:
            raise TableError(f"局号应从 1 开始: {self.round_no}")

    @classmethod
    def open(cls, dealer_seat: int = 0, round_no: int = 1) -> TableState:
        return cls(wall_remaining=INITIAL_WALL, dealer_seat=dealer_seat, round_no=round_no)

    @property
    def draws_made(self) -> int:
        return INITIAL_WALL - self.wall_remaining

    @property
    def draws_left(self) -> int:
        """本局还能摸多少张。"""
        return max(0, self.wall_remaining - RESERVED_TILES)

    @property
    def max_draws(self) -> int:
        """本局可摸总数（开局即 63 张）。"""
        return MAX_DRAWS

    @property
    def progress(self) -> float:
        """本局进度 0–1，按可摸张数消耗比例计。"""
        return min(1.0, max(0.0, 1.0 - self.draws_left / MAX_DRAWS))

    @property
    def can_draw(self) -> bool:
        return self.wall_remaining > RESERVED_TILES

    @property
    def can_gang(self) -> bool:
        return self.wall_remaining > RESERVED_TILES

    @property
    def is_exhausted(self) -> bool:
        """牌墙可摸张数耗尽，本局流局。"""
        return not self.can_draw

    def after_draw(self, count: int = 1) -> TableState:
        """取走 ``count`` 张牌（普通摸牌与杠后补牌都按 1 张计）。"""
        if count < 0:
            raise TableError(f"取牌张数不能为负: {count}")
        if count > self.draws_left:
            raise TableError(f"可摸张数不足: 需要 {count}，仅剩 {self.draws_left}")
        return TableState(
            wall_remaining=self.wall_remaining - count,
            dealer_seat=self.dealer_seat,
            round_no=self.round_no,
        )


def next_dealer_seat(
    dealer_seat: int,
    *,
    exhausted: bool,
    dealer_won: bool,
    seats: int = SEATS,
) -> int:
    """流局或庄家自摸时连庄，其余情况轮庄。"""
    if not 0 <= dealer_seat < seats:
        raise TableError(f"庄家座位越界: {dealer_seat}")
    if exhausted or dealer_won:
        return dealer_seat
    return (dealer_seat + 1) % seats


def next_round(
    table: TableState,
    *,
    exhausted: bool,
    dealer_won: bool,
    seats: int = SEATS,
) -> TableState:
    """结算一局后开启下一局。

    连庄在本平台不影响倍率（首局即按 ×8 计），因此只需跟踪庄家身份，
    无需累计连庄次数。
    """
    return TableState(
        wall_remaining=INITIAL_WALL,
        dealer_seat=next_dealer_seat(table.dealer_seat, exhausted=exhausted, dealer_won=dealer_won, seats=seats),
        round_no=table.round_no + 1,
    )
