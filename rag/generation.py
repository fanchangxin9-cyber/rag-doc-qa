"""
generation.py —— 文案

手里拿到「问题 + 几段资料」，要干两件事：
  ① 把资料编上号、按相关度升序摆好，拼成一段提示词
  ② 交给大模型写答案；写失败就重试

对外只露两个函数：make_prompt、ask。

注意：这个文件不 import knowledge，也不 import retrieval。
  —— 想看"答案是怎么拼出来的"，只看这一个文件就够了；
  —— 将来想加流式输出、多轮记忆，也主要改这里，碰不到检索那边。
"""

import math
import time

from langchain_openai import ChatOpenAI

import config


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


def make_prompt(question: str, results: list) -> tuple[str, list]:
    """把「问题 + 检索结果」拼成一段提示词，并整理出来源清单。

    收：问题，以及 [(片段, 相关度分数)] —— 最相关的排在前面
    吐：(完整提示词, 来源清单)，来源清单要原样发给前端渲染
    """
    # 搭车改动：按相关度【升序】摆放 —— 把最相关的放到最后面，紧挨着问题。
    # 依据是"近因效应"：模型对越靠近结尾的内容记得越牢。
    # results 本来是最相关在前，reversed 一下就翻过来了。
    #
    # 编号跟着摆放顺序走（编号 = 它在资料里排第几），
    # 所以「编号最大的最相关」。这条规律已经写进 config.SYSTEM_PROMPT 告诉模型了。
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

    prompt = f"{config.SYSTEM_PROMPT}\n\n资料：\n{context}问题：{question}"
    return prompt, sources


def ask(prompt: str) -> str:
    """把提示词交给大模型。

    失败就重试（等 0.5 秒、1 秒、……），重试用完仍失败才抛错。
    为什么要重试？网络抖动、服务限流这类"抽风型"故障，往往等一下自己就好了。
    """
    last_err = None
    for attempt in range(1, config.LLM_MAX_RETRY + 1):
        try:
            return LLM.invoke(prompt).content
        except Exception as e:
            last_err = e
            # 最后一次就不用再等了 —— 后面已经没有"下一次"
            if attempt < config.LLM_MAX_RETRY:
                time.sleep(config.LLM_RETRY_WAIT * attempt)

    raise GenerationError(f"大模型调用失败，已重试 {config.LLM_MAX_RETRY} 次：{last_err}")
