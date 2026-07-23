#!/usr/bin/env python3
"""voice_mcp · 给 Claude Code 用的语音信箱拉取工具

Claude Code 主动调用这里的工具去读语音信箱，不是被推送过来的。
不管是本地窗口还是 CCR 线上会话，你在哪个窗口发指令，就是哪个窗口在读，
不需要额外的"发给哪个窗口"的路由逻辑。

用 streamable-http 传输（CCR 环境下手写 HTTP MCP 不可见，必须用这个），
Bearer token 鉴权，跟 hervoice.py 共用同一个 SQLite 数据库（storage.py）。

配置：MCP_TOKEN（必填，没有就拒绝启动）、MCP_HOST、MCP_PORT、HERVOICE_DATA。
"""
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from starlette.responses import PlainTextResponse

import storage

load_dotenv()

DATA_DIR = Path(os.environ.get("HERVOICE_DATA", "./data"))
MCP_TOKEN = os.environ.get("MCP_TOKEN", "")
HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8020"))

storage.init(DATA_DIR)

mcp = FastMCP("hervoice")


@mcp.tool()
def get_unread_voice_messages(mark_read: bool = True) -> list[dict]:
    """拉取所有还没读过的语音消息（转写文字 + 语气分析），默认拉取后标记为已读。
    每条消息包含：id（编号）、ts（时间戳）、text（转写文字）、emotion（情感分类）、
    confidence（置信度）、hint（语气解读一句话）、features（声学特征：音高/能量/停顿/语速）、
    replies（之前已经追加过的回复列表，每条带自己的时间戳）。"""
    msgs = storage.get_unread(mark_read=mark_read)
    storage.log_activity("get_unread_voice_messages",
                          f"拉取未读语音 {len(msgs)} 条，id={[m['id'] for m in msgs]}")
    return msgs


@mcp.tool()
def get_recent_voice_messages(n: int = 5) -> list[dict]:
    """看最近 n 条语音消息，不管有没有读过，调用这个不会改变已读/未读状态。"""
    msgs = storage.get_recent(n)
    storage.log_activity("get_recent_voice_messages", f"查看最近 {len(msgs)} 条语音")
    return msgs


@mcp.tool()
def get_voice_messages_by_date(date: str) -> list[dict]:
    """按日期查语音消息，date 格式 'YYYY-MM-DD'，例如 '2026-07-23'。"""
    msgs = storage.get_by_date(date)
    storage.log_activity("get_voice_messages_by_date", f"按日期 {date} 查到 {len(msgs)} 条")
    return msgs


@mcp.tool()
def get_voice_message(message_id: int) -> dict | None:
    """按编号查一条语音消息的完整内容，包含它下面所有的历史回复。查不到返回 null。"""
    msg = storage.get_by_id(message_id)
    storage.log_activity("get_voice_message",
                          f"查看消息 #{message_id}" + ("" if msg else "（不存在）"))
    return msg


@mcp.tool()
def reply_to_voice_message(message_id: int, text: str) -> dict:
    """在某条语音消息下面追加一条带时间戳的回复/想法，可以对同一条消息反复调用继续追加，
    形成一条随时间增长的对话串。消息编号不存在时返回 {"ok": false}。"""
    reply_id = storage.add_reply(message_id, text)
    if reply_id is None:
        storage.log_activity("reply_to_voice_message", f"回复消息 #{message_id} 失败（不存在）")
        return {"ok": False, "error": f"message {message_id} not found"}
    storage.log_activity("reply_to_voice_message", f"给消息 #{message_id} 追加回复：{text}")
    return {"ok": True, "reply_id": reply_id}


@mcp.tool()
def mark_voice_message_read(message_id: int) -> dict:
    """手动把某条语音消息标记为已读。一般不需要手动调用——用 get_unread_voice_messages
    拉取时默认就会自动标记，这个工具是给需要单独标记某一条时用的。"""
    ok = storage.mark_read(message_id)
    storage.log_activity("mark_voice_message_read", f"标记消息 #{message_id} 已读：{ok}")
    return {"ok": ok}


@mcp.tool()
def get_recent_activity(n: int = 20) -> list[dict]:
    """查最近的操作日志——之前通过这套工具做过什么（拉取过哪些消息、回复过什么、
    标记过什么已读），带时间戳。用于失忆重开会话时找回"我最近在干嘛"的上下文。
    这个查询本身不会被记进日志（避免自我循环）。"""
    return storage.get_activity(n)


def _require_token(app):
    """极简 ASGI 中间件：streamable-http 请求必须带对的 Bearer token"""
    async def middleware(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode()
            token = auth[7:] if auth.startswith("Bearer ") else ""
            if not secrets.compare_digest(token, MCP_TOKEN):
                resp = PlainTextResponse("unauthorized", status_code=401)
                await resp(scope, receive, send)
                return
        await app(scope, receive, send)
    return middleware


if __name__ == "__main__":
    import uvicorn

    if not MCP_TOKEN:
        raise SystemExit("MCP_TOKEN 未设置，拒绝在没有鉴权的情况下启动")

    http_app = mcp.http_app()
    protected_app = _require_token(http_app)
    uvicorn.run(protected_app, host=HOST, port=PORT)
