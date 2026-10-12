"""
session.py —— 会话记忆（工具模块）

职责：按会话编号存 / 取「问 + 答」的历史，支持清空。

注意：它【不是】流水线工位，所以不受"rag 三个模块互不 import"那条铁律约束——
generation.py 可以放心用它。这是有意为之，不是破例。

将来要把记忆从「进程内存」换成「SQLite / Redis」，只改这一个文件。
"""

import config

# {会话编号: [{"role": "user"/"assistant", "content": 文字}, ...]}
# 存在进程内存里，所以改代码触发 --reload、或重启服务，都会清空。
_SESSIONS: dict[str, list] = {}


def get(session_id: str) -> list:
    """取某个会话的历史。

    太长就只留最近几条——不然聊得越久，塞给模型的内容越多、越贵、越容易跑题。
    返回的是一份**拷贝**，外面拿它怎么折腾都不会弄脏这里存的历史。
    """
    history = _SESSIONS.setdefault(session_id, [])
    if len(history) > config.MAX_HISTORY_MESSAGES:
        del history[:-config.MAX_HISTORY_MESSAGES]
    return list(history)


def append(session_id: str, question: str, answer: str) -> None:
    """把这一轮的「问 + 答」记进去。

    只记「纯问题」和「纯答案」——**检索到的资料不记**。
    因为每轮都要按当时的知识库现查现拼，把资料记下来反而会让后面几轮
    引用到过期内容、或引用了已经被删掉的文件。
    """
    _SESSIONS.setdefault(session_id, []).extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])

    # 内存底线：会话数超上限就丢掉最早的那个。
    # 字典在 Python 3.7+ 会保持"插入顺序"，所以 next(iter(...)) 就是最早建的那个。
    if len(_SESSIONS) > config.MAX_SESSIONS:
        _SESSIONS.pop(next(iter(_SESSIONS)))


def reset(session_id: str) -> None:
    """清空某个会话（页面上「新对话」按钮用）"""
    _SESSIONS.pop(session_id, None)


def clear_all() -> None:
    """清空所有会话。

    换模型的时候必须调它：历史里全是"上一个模型说过的话"，
    留给新模型看只会让它精神分裂。
    """
    _SESSIONS.clear()
