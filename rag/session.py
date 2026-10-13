"""
session.py —— 会话记忆（工具模块）

职责：按会话编号存 / 取「问 + 答」的历史，支持清空。

**存哪：本机的 SQLite 文件（config.SESSIONS_DB）。**

  为什么从"进程内存"换成它？
    内存里的东西一重启就没了 —— 改代码触发 --reload 也照样清空，
    聊到一半改行代码，之前聊过什么就全忘了。换成文件之后，重启服务、
    热重载都还在，关掉浏览器明天回来接着聊也行。

  为什么选 SQLite，不用 Redis / MySQL？
    它是 Python 【自带】的（import sqlite3 就行），不必额外装任何东西、
    不必额外开一个服务，数据就躺在项目目录下的一个文件里。
    对"本机单人使用"这个场景，它刚刚好 —— 别的都是杀鸡用牛刀。

对外接口没变，还是四个：get / append / reset / clear_all。
所以 generation.py 和 main.py 一行都不用改 —— 这就是当初把它单独拆出来的兑现。

注意：它【不是】流水线工位，所以不受"rag 各个模块互不 import"那条铁律约束——
generation.py 可以放心用它。这是有意为之，不是破例。
"""

import sqlite3
import threading
import time
from contextlib import contextmanager

import config

# 写操作排队用的锁。
# 为什么要它：FastAPI 把同步接口（def）丢进线程池跑，多个请求可能同时落库；
# 而 SQLite 同一时刻只允许一个写入者，不加锁会撞出 "database is locked"。
# 读操作不用锁 —— WAL 模式下读不阻塞写、写也不阻塞读。
_LOCK = threading.Lock()

# 两张表：
#   sessions —— 一行一个会话，只记"最后活跃时间"，用来判断该丢谁
#   messages —— 一行一条消息，问答都在这；靠自增 id 天然记住先后顺序
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _connect():
    """开一个数据库连接。

    WAL 是一种"读写互不打架"的日志模式：一边在读旧数据，另一边照样能写新的。
    不加它的话，读和写要互相等待，多人同时用时容易卡。
    """
    conn = sqlite3.connect(config.SESSIONS_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init():
    """建表（已经建过就什么都不做）。模块加载时执行一次。"""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


_init()


@contextmanager
def _db(write: bool = False):
    """开连接的统一入口：借用 → 用完一定归还，顺带提交。

    · 写操作（write=True）先拿锁排队，避免并发写撞车
    · commit 统一在这里做 —— 不用每个函数自己记着提交，忘了就白写一场
    · close 放在 finally 里 —— 哪怕中途报错，连接也一定会还回去
    """
    if write:
        _LOCK.acquire()
    try:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    finally:
        if write:
            _LOCK.release()


def get(session_id: str) -> list:
    """取某个会话的历史，形如 [{"role": "user", "content": …}, …]。

    库里每个会话最多只留 MAX_HISTORY_MESSAGES 条（见 append 里修剪那步），
    所以这里直接读就行，不用再截断。
    返回的是**新造出来的字典**，外面拿它怎么折腾都弄不脏库里的记录。
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def append(session_id: str, question: str, answer: str) -> None:
    """把这一轮的「问 + 答」记进去。

    只记「纯问题」和「纯答案」——**检索到的资料不记**。
    因为每轮都要按当时的知识库现查现拼，把资料记下来反而会让后面几轮
    引用到过期内容、或引用了已经被删掉的文件。
    """
    now = time.time()
    with _db(write=True) as conn:
        conn.executemany(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, "user", question), (session_id, "assistant", answer)],
        )

        # 记一笔"这个会话刚活跃过"。ON CONFLICT 是"已存在就更新、没存在就插入"，
        # 一句话同时处理新会话和老会话。
        conn.execute(
            "INSERT INTO sessions(session_id, updated_at) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at",
            (session_id, now),
        )

        # 修剪①：这个会话只留最近 MAX_HISTORY_MESSAGES 条。
        # 「不在最近 N 条里的，全删掉」—— 比刚才多出来的那两条一并清走，
        # 库就不会随着聊天越长越大。
        conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id NOT IN ("
            "  SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
            ")",
            (session_id, session_id, config.MAX_HISTORY_MESSAGES),
        )

        # 修剪②：会话总数超上限，丢掉【最久没用过】的那个。
        # 这里读的是 updated_at，所以丢的是"最久没来的客人"，
        # 而不是"最早建的那个" —— 天天用的会话不会被新会话挤掉。
        conn.execute(
            "DELETE FROM messages WHERE session_id NOT IN ("
            "  SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT ?"
            ")",
            (config.MAX_SESSIONS,),
        )
        conn.execute(
            "DELETE FROM sessions WHERE session_id NOT IN ("
            "  SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT ?"
            ")",
            (config.MAX_SESSIONS,),
        )


def list_sessions(limit: int = 50) -> list:
    """列出所有会话，最近用过的排前面 —— 前端拿它渲染「历史会话」列表。

    preview 取的是这个会话的【第一句提问】，而不是最后一条消息 ——
    用第一句当标题，更容易认出"这次聊的是啥"。

    为什么这个函数放在 session.py、不放在 main.py？
      因为它要读"表长什么样"，那是这个模块的内部知识。
      前台只管把它转成 JSON 发出去，不该懂 SQL。
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT s.session_id, s.updated_at, "
            "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id), "
            "       (SELECT m.content FROM messages m WHERE m.session_id = s.session_id "
            "          AND m.role = 'user' ORDER BY m.id LIMIT 1) "
            "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [
        {
            "session_id": sid,
            "updated_at": updated_at,
            "message_count": count,
            # 换行会让列表项撑高，先压成一行；太长也截断
            "preview": (preview or "").replace("\n", " ").strip()[:40],
        }
        for sid, updated_at, count, preview in rows
    ]


def reset(session_id: str) -> None:
    """清空某个会话（页面上「新对话」按钮用）"""
    with _db(write=True) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def clear_all() -> None:
    """清空所有会话。

    换模型的时候必须调它：历史里全是"上一个模型说过的话"，
    留给新模型看只会让它精神分裂。
    """
    with _db(write=True) as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")