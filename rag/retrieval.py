"""
retrieval.py —— 情报组

一次提问，情报组要干三件事：
  ① 两路人马分头找 —— 向量（看"意思像不像"）+ 关键词（看"字面对不对得上"）
  ② RRF 融合       —— 把两份排名合成一份
  ③ 重排           —— 用交叉编码器逐条细看，挑出最相关的几段

对外只露两个函数：retrieve（干活的）、warmup（提前把模型下载好）。

注意：这个文件既 import knowledge，也不 import generation。
三个模块互不往来，全部由 main.py 串起来。
"""

import jieba
from langchain_community.retrievers import BM25Retriever
from fastembed.rerank.cross_encoder import TextCrossEncoder

import config


# ============================================================
# 一、召回器：先定一个统一的"形状"
# ============================================================
class Retriever:
    """召回器的统一形状：给一个问题和一个数量 k，吐一串最相关的片段。

    想加第三种召回方式（比如"按标题匹配"）？只要写一个子类实现 search()，
    再到 retrieve() 里加一行就行 —— 融合和重排的代码完全不用动。
    这就是"接口"的价值：约定好形状，零件就能随便换。
    """

    def search(self, question: str, k: int) -> list:
        raise NotImplementedError("子类必须实现 search()")


class VectorRetriever(Retriever):
    """向量召回：看"意思像不像"。同义改写、口语化提问，它找得准。"""

    def __init__(self, db):
        self.db = db

    def search(self, question: str, k: int) -> list:
        return self.db.as_retriever(search_kwargs={"k": k}).invoke(question)


class KeywordRetriever(Retriever):
    """关键词召回：看"字面对不对得上"。型号、条款编号、专有名词，它找得准。"""

    def __init__(self, corpus: list):
        # BM25 的原理是"数词"，所以要先知道全库有哪些片段，才能建词表。
        #
        # preprocess_func=jieba.lcut 是关键：
        # BM25 默认按【空格】切词，中文没有空格，整句话会被当成一个"巨词"，
        # 于是每篇文档的词表都只有 1 个词 —— 关键词检索会彻底失效。
        self.retriever = BM25Retriever.from_documents(
            corpus,
            preprocess_func=jieba.lcut,
        )

    def search(self, question: str, k: int) -> list:
        self.retriever.k = k         # 先告诉它"要几条"
        return self.retriever.invoke(question)


# ============================================================
# 二、RRF 融合
# ============================================================
def fuse(doc_lists: list, weights: list, c: int) -> list:
    """把多路召回结果合成一份排名。

    公式：某段资料的得分 = Σ  权重 / (c + 它在这一路里的名次)

    为什么不直接把两路的分数相加？
      向量给的是 0~1 的相似度，BM25 给的是没有上界的词频分，
      两个量纲不同，加起来等于拿苹果加橘子。
      名次是所有召回方式共用的语言，所以只能加名次、不加分数。
    """
    score = {}      # 片段内容 → 累计得分
    pool = {}       # 片段内容 → 代表片段（最后用它取回 Document 对象）

    for docs, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(docs, start=1):      # 名次从 1 开始数
            key = doc.page_content
            pool[key] = doc
            score[key] = score.get(key, 0.0) + weight / (c + rank)

    # 按累计得分从高到低排。
    # "两路都排在前面"的片段分数会叠加，自然浮到最前面。
    return sorted(pool.values(), key=lambda d: score[d.page_content], reverse=True)


# ============================================================
# 三、重排
# ============================================================
_reranker = None            # 模型对象（第一次用到时才下载）
_reranker_failed = False    # 记一笔"加载失败"，免得每次提问都重新尝试下载


def _get_reranker():
    """懒加载重排模型。加载失败就返回 None，调用方自动降级，服务不会挂。"""
    global _reranker, _reranker_failed
    if _reranker is not None or _reranker_failed:
        return _reranker

    try:
        print(f"正在加载重排模型 {config.RERANK_MODEL}（首次运行需下载约 1GB，请耐心等待）…")
        _reranker = TextCrossEncoder(model_name=config.RERANK_MODEL, cache_dir=config.CACHE_DIR)
        print("重排模型就绪")
    except Exception as e:
        print(f"[警告] 重排模型加载失败，本次将退回纯向量检索：{e}")
        _reranker_failed = True

    return _reranker


def warmup():
    """让 main.py 在服务启动时调用一次。

    作用：把 1GB 的模型下载提前做掉，免得用户第一次提问时
    盯着"思考中…"以为是卡死了。
    """
    _get_reranker()


def _rerank(question: str, docs: list) -> tuple[list, bool]:
    """用交叉编码器逐条给「问题 × 片段」打分，再按分数从高到低排。

    返回：(排好序的 [(片段, 分数)], 这次到底有没有真的重排)
    模型不可用时，分数一律给 None，顺序保持原样。
    """
    reranker = _get_reranker()
    if reranker is None or not docs:
        return [(d, None) for d in docs], False

    scores = list(reranker.rerank(question, [d.page_content for d in docs]))
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [(d, float(s)) for d, s in ranked], True


# ============================================================
# 四、对外入口
# ============================================================
def retrieve(db, question: str, corpus: list) -> tuple[list, bool]:
    """情报组的完整工作流程。

    收：向量库 + 问题 + 全库片段（关键词检索建词表要用）
    吐：([(片段, 相关度分数)], 是否用上了重排)
        列表按相关度从高到低排；分数为 None 表示这次没重排。
    """
    if not corpus:
        return [], False

    # ① 两路人马分头找，各自先粗捞一批
    vector_docs = VectorRetriever(db).search(question, config.VECTOR_K)
    keyword_docs = KeywordRetriever(corpus).search(question, config.KEYWORD_K)

    # ② RRF 融合：把两份排名合成一份
    candidates = fuse([vector_docs, keyword_docs], config.RRF_WEIGHTS, config.RRF_C)

    # ③ 重排：逐条细看，挑出最相关的前几段
    ranked, rerank_used = _rerank(question, candidates)
    return ranked[:config.RERANK_TOP], rerank_used
