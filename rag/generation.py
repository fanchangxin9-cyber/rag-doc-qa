"""
generation.py —— 文案

手里拿到「问题 + 几段资料」，要干三件事：
  ① 把资料编上号、按相关度升序摆好            → build_context（纯函数）
  ② 把「系统规则 + 历史问答 + 本轮资料问题」组起来 → build_messages（纯函数）
  ③ 交给大模型，边生成边往外吐，并记住这一轮    → ask_stream（生成器）

对外只露三个函数：build_context / build_messages / ask_stream。

注意：它只 import session（会话记忆这个"工具"）和 config，
不 import knowledge，也不 import retrieval。
  —— 想看"答案是怎么拼出来的"，只看这一个文件就够了；
  —— 将来想加别的呈现方式，也主要改这里，碰不到检索那边。
"""

import math
import time
from itertools import chain

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import config
from rag import session


class GenerationError(Exception):
    """大模型环节「能被预料到」的错误：重试若干次仍然失败。"""


# 大模型：文案自己的工具，负责"把资料写成通顺的答案"。
# 模型名、密钥、服务地址全部从 config 读 —— 换模型只改 config 一行。
#
# 小改进：不写 os.environ["OPENAI_API_KEY"]，而是直接把密钥传给构造函数的参数。
# 好处是不去污染整个程序的全局环境变量，密钥只在这一处出现。
LLM = ChatOpenAI(
    model=config.LLM_MODEL,
    openai_api_key=config.LLM_API_KEY,
    openai_api_base=config.LLM_BASE_URL,
)


def _score_to_ratio(logit: float) -> float:
    """把重排模型的原始打分压进 0~1，变成人看得懂的"相关度"。

    重排模型吐的是没有上下界的原始分（logit）——可能 -15，也可能 +8。
    界面上直接写「相关度 -10.20」，用户会以为系统坏了；其实它只是想说
    "这段最不相关"。

    套一个 sigmoid（S 形函数）就好了：正数 → 接近 1，负数 → 接近 0，0 → 0.5。
    """
    return round(1 / (1 + math.exp(-logit)), 4)


# ============================================================
# 一、纯函数：拼资料、组消息
#     —— 不联网、不写文件，只靠"进来的参数"算结果，所以最好测
# ============================================================
def build_context(results: list) -> tuple[str, list]:
    """把 [(片段, 相关度分数)] 拼成带编号的资料文本 + 来源清单。

    收：最相关的排在前面
    吐：(资料文本, 来源清单)，来源清单要原样发给前端渲染

    两个关键约定：
      · 搭车改动：按相关度【升序】摆放 —— 最相关的放到最后面，紧挨着问题。
        依据是"近因效应"：模型对越靠近结尾的内容记得越牢。
      · 编号跟着摆放顺序走（编号 = 它在资料里排第几），
        所以「编号最大的最相关」。这条规律已经写进 config.SYSTEM_PROMPT 告诉模型了。
    """
    ordered = list(reversed(results))

    context = ""
    sources = []
    for i, (doc, score) in enumerate(ordered, start=1):
        file_name = doc.metadata.get("file_name", "未知文件")
        page = doc.metadata.get("page", 0) + 1      # 页码从 0 开始数，给人看要 +1
        context += f"[{i}]（{file_name}·第 {page} 页）\n{doc.page_content}\n\n"
        sources.append({
            "index": i,
            "file": file_name,
            "page": page,
            "score": _score_to_ratio(score) if score is not None else None,
            "text": doc.page_content,
        })

    return context, sources


def build_messages(question: str, context: str, history: list) -> list:
    """组装「带角色」的消息列表 —— 这是多轮记忆的载体。

    结构：[系统规则] + [历史的一问一答…] + [本轮：资料 + 问题]

    为什么必须用消息列表、不能像以前那样拼成一个字符串？
    因为「角色」（system / user / assistant）是模型区分"谁说的话"的唯一依据。
    全塞进一个字符串，模型就分不清哪句话是它自己上一轮说过的。
    """
    messages = [SystemMessage(content=config.SYSTEM_PROMPT)]

    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))

    messages.append(HumanMessage(content=f"资料：\n{context}问题：{question}"))
    return messages


# ============================================================
# 二、带副作用的：建立流（含重试）、边吐边记
# ============================================================
def _open_stream(messages: list):
    """建立流，并"逼"它立刻吐出第一块。

    为什么不能只写 LLM.stream(messages) 就完事？
    因为那是个"懒"生成器 —— 光调用它不会真正联网，直到你开始取数据才连。
    所以必须立刻 next() 一下，把"连不上 / 超时 / 密钥错"这类问题提前逼出来。

    为什么重试只能在这一步做？
    因为一旦已经有字发给用户了，重来一遍会把内容重复输出。
    流式下没法"整段重试"，这是流式的固有取舍。
    """
    last_err = None
    for attempt in range(1, config.LLM_MAX_RETRY + 1):
        try:
            stream = LLM.stream(messages)
            first = next(stream, None)      # next(..., None)：流是空的也不报错，给个 None
            return stream, first
        except Exception as e:
            last_err = e
            # 最后一次就不用再等了 —— 后面已经没有"下一次"
            if attempt < config.LLM_MAX_RETRY:
                time.sleep(config.LLM_RETRY_WAIT * attempt)

    raise GenerationError(f"大模型调用失败，已重试 {config.LLM_MAX_RETRY} 次：{last_err}")


def ask_stream(session_id: str, question: str, context: str):
    """流式生成答案，并在结束时把这一轮记进会话记忆。一次 yield 一小段文字。

    注意：这是个【生成器】—— 调用它不会立刻跑完，而是每 yield 一次就交给上层一次。
    上层（main.py）负责把每一小段包成 SSE 推给浏览器。
    """
    history = session.get(session_id)               # 取出这个会话之前的问答
    messages = build_messages(question, context, history)

    stream, first = _open_stream(messages)

    answer = ""
    # chain([第一块], 剩下的流) = 把两段接成一段，这样刚才那一块不用丢掉重来
    for chunk in chain([first], stream):
        if chunk is None:                           # 空流的情况
            continue
        piece = chunk.content
        if piece:
            answer += piece
            yield piece                             # ← 吐给上层

    # 只有完整生成完（没有中途报错）才记进记忆 —— 半截答案记进去只会污染上下文
    if answer:
        session.append(session_id, question, answer)
