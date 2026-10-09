"""网页对局服务（零外部依赖，仅 Python 标准库）。

路由
----
* ``GET  /``                  → 前端页面（webapp/dist/index.html，构建产物）
* ``GET  /assets/*``          → 前端静态资源
* ``GET  /api/state``         → 当前对局状态快照 JSON
* ``GET  /api/stream``        → SSE 事件流（state / action_required / round_end / match_end）
* ``POST /api/action``        → 提交人类动作 {kind, tile, tiles, gang_kind}
* ``POST /api/new``           → 开新一场 {rounds?, seed?, human_seat?, mode?, decider?}
* ``POST /api/report``        → 报障：保存当前局面 + 决策建议 + 备注为 JSON 快照 {comment}

运行::

    .venv/bin/python webapp/server.py --port 8848
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from session import GameSession, DEFAULT_DECIDER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")

_session: GameSession | None = None
_session_lock = threading.Lock()


def get_session() -> GameSession:
    global _session
    with _session_lock:
        if _session is None:
            _session = GameSession(rounds=8, seed=None, decider=DEFAULT_DECIDER)
            _session.start()
        return _session


def new_session(**kwargs) -> GameSession:
    global _session
    with _session_lock:
        _session = GameSession(**kwargs)
        _session.start()
        return _session


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- 工具 ------------------------------------------------------------- #
    def _json(self, obj, status=200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def log_message(self, fmt, *args) -> None:  # 降噪
        pass

    # -- 路由 ------------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/state":
            sess = get_session()
            self._json(
                {
                    "status": sess.status,
                    "finished": sess.finished,
                    "state": sess.current_state,
                    "rounds": sess.rounds_played,
                    "cumulative": sess._cumulative,
                    "human_seat": sess.human_seat,
                    "total_rounds": sess.rounds,
                    "reveal": sess.reveal,
                    # 刷新/重连：把当前待人类决策的事件一并返回，前端可立即恢复按钮
                    "required": sess._last_required,
                    # 建议：开启状态 + 最近一次建议（重连后恢复显示）
                    "suggest_enabled": sess.suggest_enabled,
                    "suggestion": sess._last_suggestion,
                    # 当前决策档（v7=线上采集器档；界面标题/建议栏据此显示）
                    "decider": getattr(sess, "decider", DEFAULT_DECIDER),
                    # 手动续局：是否停在上局结束、等点「开下一局」
                    "awaiting_next": sess.awaiting_next,
                }
            )
            return
        if path == "/api/stream":
            self._sse()
            return
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/action":
            sess = get_session()
            ok, err = sess.submit(self._read_json())
            self._json({"ok": ok, "error": err}, 200 if ok else 400)
            return
        if path == "/api/new":
            body = self._read_json()
            kwargs = {}
            if body.get("rounds"):
                kwargs["rounds"] = int(body["rounds"])
            if body.get("seed") is not None and body["seed"] != "":
                kwargs["seed"] = int(body["seed"])
            if body.get("human_seat") is not None:
                kwargs["human_seat"] = int(body["human_seat"])
            if body.get("mode"):
                kwargs["mode"] = str(body["mode"])
            if body.get("ai_delay") is not None:
                kwargs["ai_delay"] = float(body["ai_delay"])
            if body.get("decider"):
                kwargs["decider"] = str(body["decider"])
            # 原地重开：保留同一事件总线，SSE 连接不断（否则浏览器收不到新对局事件）。
            sess = get_session()
            sess.reset(**kwargs)
            self._json({"ok": True, "status": sess.status})
            return
        if path == "/api/reveal":
            body = self._read_json()
            sess = get_session()
            sess.set_reveal(bool(body.get("reveal")))
            self._json({"ok": True, "reveal": sess.reveal})
            return
        if path == "/api/suggest":
            body = self._read_json()
            sess = get_session()
            sess.set_suggest(bool(body.get("enabled")))
            self._json({"ok": True, "suggest_enabled": sess.suggest_enabled})
            return
        if path == "/api/next_round":
            sess = get_session()
            ok, err = sess.next_round()
            self._json({"ok": ok, "error": err, "awaiting_next": sess.awaiting_next},
                       200 if ok else 409)
            return
        if path == "/api/report":
            body = self._read_json()
            sess = get_session()
            report = sess.snapshot_report(body.get("comment") or "")
            self._json({"ok": True, "id": report.get("id"), "path": report.get("path"),
                        "created_at": report.get("created_at")})
            return
        self._json({"error": "not found"}, 404)

    # -- 静态文件 --------------------------------------------------------- #
    def _static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        safe = os.path.normpath(path).lstrip("/")
        full = os.path.join(DIST, safe)
        if not full.startswith(DIST) or not os.path.isfile(full):
            self._json({"error": "not found", "hint": "先构建前端: cd webapp && npm run build"}, 404)
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 禁止浏览器缓存：前端重新构建后普通刷新即生效，
        # 避免加载到旧构建产物（前端有修复却看不到“仍要刷新”的根因）。
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    # -- SSE -------------------------------------------------------------- #
    def _sse(self) -> None:
        sess = get_session()
        q = sess.bus.subscribe()
        # 刷新/重连补发：若当前正等人类决策，把该事件也推入队列，
        # 否则重连后只会重放最后一条（可能是 state），按钮就丢了。
        last_required = sess._last_required
        if last_required is not None and sess.bus.last is not last_required:
            try:
                q.put_nowait(last_required)
            except queue.Full:
                pass
        # 同理补发「最近一次建议」，重连后建议卡片不丢失。
        last_suggestion = sess._last_suggestion
        if last_suggestion is not None:
            try:
                q.put_nowait(last_suggestion)
            except queue.Full:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            sess.bus.unsubscribe(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--human-seat", type=int, default=0)
    ap.add_argument("--ai-delay", type=float, default=2.0,
                    help="AI 主动出牌前的思考停顿秒数（默认 2）")
    ap.add_argument("--decider", default=DEFAULT_DECIDER,
                    help=f"决策档（默认 {DEFAULT_DECIDER}，与线上采集器一致；可换 v5/v6 等）")
    args = ap.parse_args()

    new_session(rounds=args.rounds, seed=args.seed, human_seat=args.human_seat,
                ai_delay=args.ai_delay, decider=args.decider)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"杭州麻将网页对局: http://{args.host}:{args.port}/  (Ctrl-C 退出)")
    print(f"静态资源目录: {DIST}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出")
        srv.shutdown()


if __name__ == "__main__":
    main()
