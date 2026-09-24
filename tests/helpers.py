"""测试用的手牌构造辅助。"""

from __future__ import annotations

from majiang.rules import tiles


def parse_spec(spec: str) -> list[int]:
    """把 ``"1w2w3w东东东白白"`` 这类紧凑写法解析为牌索引列表。"""
    codes: list[str] = []
    pending = ""
    for char in spec:
        if char.isspace():
            continue
        if pending:
            codes.append(pending + char)
            pending = ""
        elif char.isdigit():
            pending = char
        else:
            codes.append(char)
    assert not pending, f"规格结尾有多余数字: {spec!r}"
    return tiles.parse_all(codes)


def counts_of(spec: str) -> list[int]:
    return tiles.counts_from(parse_spec(spec))
