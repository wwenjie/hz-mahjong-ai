#!/usr/bin/env python
"""赛时到位看护（保险网）。

引擎自身会在 status∈{registering,stage_open} 时到位；本脚本是独立备份：
每 8 秒轮询赛事状态，一旦出现可到位窗口就立即调用两个到位端点（均幂等），
把每一次状态变迁与到位结果落盘到 logs/ready_watch.log。

用法（脱离会话运行）：
    set -a && source ~/.majiang/env && set +a
    setsid nohup .venv/bin/python tools/tournament_ready_watch.py <tournament_id> \
        >> logs/ready_watch.out 2>&1 < /dev/null &
"""
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, "src")
from majiang.client.transport import HttpTransport  # noqa: E402
from majiang.client.errors import ApiError  # noqa: E402

SERVER = os.environ["MAJIANG_SERVER"]
TOK = os.environ["MAJIANG_TOKEN_MAIN"]
TID = sys.argv[1] if len(sys.argv) > 1 else "t_e3c195576228"
READY_WINDOW = {"registering", "stage_open"}


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def main() -> None:
    t = HttpTransport(SERVER)
    last = None
    log(f"看护启动 tid={TID}")
    while True:
        try:
            st = t.request("GET", f"/api/tournaments/{TID}", token=TOK)
        except ApiError as e:
            log(f"状态查询 ApiError: {e.code}")
            time.sleep(8)
            continue
        except Exception as e:  # noqa: BLE001
            log(f"状态查询异常: {type(e).__name__} {e}")
            time.sleep(8)
            continue

        status = st.get("status")
        stage = (st.get("stage") or {}).get("no")
        key = (status, st.get("stage_status"), st.get("stage_crashed"), st.get("voided_reason"))
        if key != last:
            log(f"状态变迁 -> status={status} stage={stage} stage_status={st.get('stage_status')} "
                f"voided={st.get('voided_reason')} ready={st.get('ready_users')}/{st.get('registered_users')}")
            last = key

        if status in READY_WINDOW:
            for path in (f"/api/tournaments/{TID}/ready", "/api/tournaments/me/ready"):
                try:
                    t.request("POST", path, token=TOK)
                    log(f"到位成功 {path}")
                except ApiError as e:
                    log(f"到位被拒 {path}: {e.code}")
                except Exception as e:  # noqa: BLE001
                    log(f"到位异常 {path}: {type(e).__name__} {e}")

        # 终态退出
        if status in {"finished", "closed", "cancelled", "canceled"}:
            log(f"赛事终态 {status}，看护退出")
            return

        time.sleep(8)


if __name__ == "__main__":
    main()
