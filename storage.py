"""SQLite 存储层：语音消息 + 回复，web 端和 MCP server 共用同一份数据。

每条语音消息是一条永久记录（不会被覆盖/轮换），带自增 id、时间戳、
已读/未读状态；replies 记录 Claude Code 追加的带时间戳回复，可以反复追加。
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH: Path | None = None


def init(data_dir: Path):
    global DB_PATH
    data_dir.mkdir(parents=True, exist_ok=True)
    DB_PATH = data_dir / "hervoice.db"
    with _connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            text TEXT NOT NULL,
            emotion TEXT,
            confidence REAL,
            hint TEXT,
            features TEXT,
            audio TEXT,
            read INTEGER NOT NULL DEFAULT 0
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL REFERENCES messages(id),
            ts TEXT NOT NULL,
            text TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT NOT NULL
        )""")


@contextmanager
def _connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["features"] = json.loads(d["features"]) if d["features"] else {}
    d["read"] = bool(d["read"])
    return d


def add_message(text, emotion, confidence, hint, features, audio="") -> int:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO messages (ts,text,emotion,confidence,hint,features,audio,read) "
            "VALUES (?,?,?,?,?,?,?,0)",
            (ts, text, emotion, confidence, hint,
             json.dumps(features, ensure_ascii=False), audio))
        return cur.lastrowid


def get_recent(n: int = 10) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    out = [_row_to_dict(r) for r in rows]
    for m in out:
        m["replies"] = get_replies(m["id"])
    return out


def get_messages_page(page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
    """给网页语音信箱用：分页返回全部历史记录，(本页消息, 总条数)。"""
    offset = max(page - 1, 0) * page_size
    with _connect() as con:
        total = con.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        rows = con.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ? OFFSET ?",
            (page_size, offset)).fetchall()
    out = [_row_to_dict(r) for r in rows]
    for m in out:
        m["replies"] = get_replies(m["id"])
    return out, total


def search_messages_page(keyword: str, page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
    """按关键词搜语音信箱（匹配转写文字或语气解读），分页返回 (本页消息, 总命中数)。"""
    like = f"%{keyword}%"
    offset = max(page - 1, 0) * page_size
    with _connect() as con:
        total = con.execute(
            "SELECT COUNT(*) c FROM messages WHERE text LIKE ? OR hint LIKE ?",
            (like, like)).fetchone()["c"]
        rows = con.execute(
            "SELECT * FROM messages WHERE text LIKE ? OR hint LIKE ? "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (like, like, page_size, offset)).fetchall()
    out = [_row_to_dict(r) for r in rows]
    for m in out:
        m["replies"] = get_replies(m["id"])
    return out, total


def get_unread(mark_read: bool = True) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM messages WHERE read=0 ORDER BY id ASC").fetchall()
        if mark_read and rows:
            con.executemany("UPDATE messages SET read=1 WHERE id=?",
                             [(r["id"],) for r in rows])
    out = [_row_to_dict(r) for r in rows]
    for m in out:
        m["replies"] = get_replies(m["id"])
    return out


def delete_message(msg_id: int) -> bool:
    """删掉一条语音消息，连带它下面的回复一起删（没有外键级联，手动清）。"""
    with _connect() as con:
        con.execute("DELETE FROM replies WHERE message_id=?", (msg_id,))
        cur = con.execute("DELETE FROM messages WHERE id=?", (msg_id,))
        return cur.rowcount > 0


def get_by_date(date_str: str) -> list[dict]:
    """date_str: 'YYYY-MM-DD'"""
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM messages WHERE ts LIKE ? ORDER BY id ASC",
            (date_str + "%",)).fetchall()
    out = [_row_to_dict(r) for r in rows]
    for m in out:
        m["replies"] = get_replies(m["id"])
    return out


def get_by_id(msg_id: int) -> dict | None:
    with _connect() as con:
        row = con.execute("SELECT * FROM messages WHERE id=?", (msg_id,)).fetchone()
    if not row:
        return None
    m = _row_to_dict(row)
    m["replies"] = get_replies(msg_id)
    return m


def update_text(msg_id: int, text: str) -> bool:
    """只改转写文字，不碰情感/语气分析结果——那是录音当下的真实记录，事后不能改。"""
    with _connect() as con:
        cur = con.execute("UPDATE messages SET text=? WHERE id=?", (text, msg_id))
        return cur.rowcount > 0


def mark_read(msg_id: int) -> bool:
    with _connect() as con:
        cur = con.execute("UPDATE messages SET read=1 WHERE id=?", (msg_id,))
        return cur.rowcount > 0


def add_reply(message_id: int, text: str) -> int | None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as con:
        exists = con.execute(
            "SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone()
        if not exists:
            return None
        cur = con.execute(
            "INSERT INTO replies (message_id,ts,text) VALUES (?,?,?)",
            (message_id, ts, text))
        return cur.lastrowid


def get_replies(message_id: int) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT ts,text FROM replies WHERE message_id=? ORDER BY id ASC",
            (message_id,)).fetchall()
    return [dict(r) for r in rows]


def log_activity(action: str, detail: str) -> int:
    """记一笔操作日志：通过 MCP 做了什么，人和 AI 都能回头查。"""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO activity_log (ts,action,detail) VALUES (?,?,?)",
            (ts, action, detail))
        return cur.lastrowid


def get_activity(n: int = 20) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]


def get_activity_page(page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
    """给网页操作日志用：分页返回全部历史记录，(本页记录, 总条数)。"""
    offset = max(page - 1, 0) * page_size
    with _connect() as con:
        total = con.execute("SELECT COUNT(*) c FROM activity_log").fetchone()["c"]
        rows = con.execute(
            "SELECT * FROM activity_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (page_size, offset)).fetchall()
    return [dict(r) for r in rows], total
