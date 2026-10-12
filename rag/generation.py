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
from rag import session, settings


class GenerationError(Exception):
    """大模型环节「能被预料到」的错误：重试若干次仍然失败。"""


# ============================================================
# 大模型对象：按需构造，不再是"启动就定死"的常量
# ============================================================
def make_llm(base_url: str, model: str, api_key: str):
    """造一个大模型对象。get_llm() 和网页上的"测试连接"都用它。"""
    return ChatOpenAI(
        model=model,
        # 本地 Ollama 这种不需要密钥的服务，给个占位值就行（不少客户端要求非空）
        openai_api_key=api_key or "not-needed",
        openai_api_base=base_url,
        # 让"流式"返回也带上 token 用量。实测支持，能拿到真实数字。
        stream_usage=True,
    )


# 缓存：同一个设置只建一次对象，不用每次提问都新建
_llm_cache = {"key": None, "llm": None}


def get_llm():
    """按「网页上当前选中那一家」的配置拿一个大模型对象；配置一变就自动重建。

    为什么不能像以前那样写 `LLM = ChatOpenAI(...)`？
    因为那是【模块级常量】—— 启动那一刻就定死了，运行中改不动。
    想在网页上换模型，就必须改成"每次现查配置、需要时重建"。
    """
    s = settings.get()
    if s is None:
        raise GenerationError("还没有配置大模型，请到页面下方的「模型设置」里选一家并填好密钥")

    key = (s["base_url"], s["model"], s["api_key"])
    if _llm_cache["key"] != key:
        _llm_cache["key"] = key
        _llm_cache["llm"] = make_llm(s["base_url"], s["model"], s["api_key"])
    return _llm_cache["llm"]


def _score_to_ratio(logit: float) -> float:
    """把单条原始分（logit）压进 0~1：正数 → 接近 1，负数 → 接近 0。

    套一个 sigmoid（S 形函数）就好了：正数 → 接近 1，负数 → 接近 0，0 → 0.5。

    现在它只在两种「退化情况」下出场：本次只有一条候选、或所有候选分数一样 ——
    那时没有"高低"可言，只能看这条资料【本身】够不够相关。见 _ratios()。
    """
    return round(1 / (1 + math.exp(-logit)), 4)


def _ratios(raw: list) -> list:
    """把「本次召回的一组原始分」整组映射成 0~1 的相关度，保持排序不变。

    为什么不能像以前那样，一条条套 sigmoid 就完事？
      重排模型吐的是【无上下界】的 logit：相关的约 +1~+3，不相关的挤在 -5~-10。
      sigmoid 会把 -5 以下全部压成 0.000x，界面取两位小数就全是 0.00 ——
      一次回答给 5 条来源、通常只有 1 条真相关，于是 4 条都显示 0，看着像坏了。
      实测下来排序是准的，坏的只是"刻度没铺开"。

    所以改成两步相乘：
      ① 铺刻度：(本条 − 最低) / (最高 − 最低)  → 最低 0、最高 1，中间按比例铺满
      ② 乘信心：sigmoid(本次最高分)             → 本次整体"到底有没有真相关"
    ① 保证永远看得出「谁更相关」；② 保证在问"今天天气"这种全不相关的问题时，
    不会把最高那条硬说成 1.00（那样就是在骗人）。
    """
    hi, lo = max(raw), min(raw)
    if hi == lo:
        # 所有分数一样（含只有一条候选）：没有相对高低，退回"看它本身够不够相关"
        return [_score_to_ratio(s) for s in raw]

    conf = _score_to_ratio(hi)                 # 本次召回的"整体信心"
    return [round((s - lo) / (hi - lo) * conf, 4) for s in raw]


def _cost(input_tokens: int, output_tokens: int) -> float:
    """按 config 里的单价，算这次大约花了多少钱（单位：元）。

    单价是"每百万 token 多少钱"，所以要除以 100 万。
    """
    return (input_tokens * config.PRICE_INPUT_PER_M
            + output_tokens * config.PRICE_OUTPUT_PER_M) / 1_000_000


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
      · 相关度必须【整组一起算】才有高低可言，所以先算好 _ratios，
        再按编号一条条装配 —— 单条是算不出"相对"的。
    """
    ordered = list(reversed(results))

    raw = [score for _, score in ordered]
    ratios = _ratios(raw) if all(s is not None for s in raw) else [None] * len(raw)

    context = ""
    sources = []
    for idx, (doc, score) in enumerate(ordered):
        i = idx + 1
        file_name = doc.metadata.get("file_name", "未知文件")
        # location 是一句由「档案室」填好的位置说明：
        #   PDF 是「第 3 页」，Word/Markdown 是「一、概述 > 1.2 安装」，TXT 是「第 3–7 段」。
        # 文案层只认这句话，【不知道】文件原本是什么格式 —— 再加新格式也不用改这里。
        location = doc.metadata.get("location", "")
        label = f"{file_name}·{location}" if location else file_name
        context += f"[{i}]（{label}）\n{doc.page_content}\n\n"
        sources.append({
            "index": i,
            "file": file_name,
            "location": location,
            "score": ratios[idx],          # None = 本次没跑重排（前端显示「未重排」）
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
            stream = get_llm().stream(messages)
            first = next(stream, None)      # next(..., None)：流是空的也不报错，给个 None
            return stream, first
        except Exception as e:
            last_err = e
            # 最后一次就不用再等了 —— 后面已经没有"下一次"
            if attempt < config.LLM_MAX_RETRY:
                time.sleep(config.LLM_RETRY_WAIT * attempt)

    raise GenerationError(f"大模型调用失败，已重试 {config.LLM_MAX_RETRY} 次：{last_err}")


def _fill_stats(stats: dict, messages: list, answer: str, last_chunk) -> None:
    """把这一轮的用量和花费装进 stats，并在终端打一行。

    优先用 API 返回的**真实用量**；万一拿不到，就按字符数粗估，并标记 estimated=True
    （宁可标清楚"这是估算"，也不假装精确）。
    """
    usage = getattr(last_chunk, "usage_metadata", None) if last_chunk is not None else None

    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        estimated = False
    else:
        # 兜底估算：prompt 的字符数已经把「系统提示词 + 历史 + 资料 + 问题」全算进去了
        prompt_chars = sum(len(m.content) for m in messages)
        input_tokens = int(prompt_chars * config.TOKENS_PER_CHAR)
        output_tokens = int(len(answer) * config.TOKENS_PER_CHAR)
        estimated = True

    cost = _cost(input_tokens, output_tokens)
    stats.update(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=round(cost, 6),
        estimated=estimated,
    )

    print(f"[用量] 输入 {input_tokens} · 输出 {output_tokens} tokens "
          f"· 约 ¥{cost:.4f}" + ("（估算值）" if estimated else ""))


def context_status(session_id: str) -> dict:
    """当前会话的「记忆占用」情况，给前端做上下文提醒用。

    full=True 表示记忆已经装满 —— 再聊下去，最早的那一轮就会被挤出去。
    """
    used = len(session.get(session_id))
    return {
        "used": used,
        "max": config.MAX_HISTORY_MESSAGES,
        "full": used >= config.MAX_HISTORY_MESSAGES,
    }


def ask_stream(session_id: str, question: str, context: str, stats: dict):
    """流式生成答案；结束时把这一轮记进会话记忆，并把用量写进 stats。

    注意：这是个【生成器】—— 调用它不会立刻跑完，而是每 yield 一次就交给上层一次。
    上层（main.py）负责把每一小段包成 SSE 推给浏览器。

    stats 是由调用方传进来的一个**空字典**（相当于"递进来一个空盒子"），
    函数跑完会往里装：{"input_tokens":…, "output_tokens":…, "cost":…, "estimated":…}
    这样就不必为了"多返回一个数"去改生成器一次只吐一小段的形状。
    """
    history = session.get(session_id)               # 取出这个会话之前的问答
    messages = build_messages(question, context, history)

    stream, first = _open_stream(messages)

    answer = ""
    last_chunk = None
    # chain([第一块], 剩下的流) = 把两段接成一段，这样刚才那一块不用丢掉重来
    for chunk in chain([first], stream):
        if chunk is None:                           # 空流的情况
            continue
        last_chunk = chunk                          # 留着它 —— 用量挂在"最后一块"上
        piece = chunk.content
        if piece:
            answer += piece
            yield piece                             # ← 吐给上层

    _fill_stats(stats, messages, answer, last_chunk)

    # 只有完整生成完（没有中途报错）才记进记忆 —— 半截答案记进去只会污染上下文
    if answer:
        session.append(session_id, question, answer)
