"""
knowledge.py —— 档案室

职责只有三件：
  1. 文件怎么进来（解析 PDF → 切块）
  2. 怎么存（写进向量库）
  3. 怎么查、怎么删

对外只露四个函数：load_db / add_file / list_files / delete_file。

注意：这个文件刻意不 import fastapi。
档案室不该知道"网页"这个东西存在 —— 这样它将来也能被命令行工具、
测试脚本直接调用。出错时它抛 KnowledgeError，由 main.py 负责翻译成
给浏览器看的错误。
"""

import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

import config


class KnowledgeError(Exception):
    """知识库相关的「能被预料到」的错误。

    比如：文件读不出文字、重复上传、要删的文件不存在。
    能被预料的错误，用这个异常明确说出来；
    真正没预料到的 bug（代码写错了）就让它原样炸掉，别在这里吞掉。
    """


# 向量模型：档案室自己的工具，负责"把文字变成一串数字"。
# 每个模块自己养自己需要的模型，互不干扰。
EMBEDDING = FastEmbedEmbeddings(model_name=config.EMBED_MODEL, cache_dir=config.CACHE_DIR)


def load_db():
    """取出向量库。库里还是空的，就返回 None（由调用方决定怎么提示）"""
    if not os.path.exists(config.DB_PATH):
        return None
    return FAISS.load_local(
        config.DB_PATH,
        EMBEDDING,
        # FAISS 用 pickle 存盘，加载时必须显式同意，否则会报错。
        # 因为库文件是我们自己生成的、不来自外部，这里同意是安全的。
        allow_dangerous_deserialization=True,
    )


def all_chunks(db):
    """把库里所有片段取出来。

    为什么要这么绕？因为 FAISS 没有"列出全部文档"的公开接口，
    只能从内部的 docstore 字典里捞（`_dict` 前面的下划线就表示"内部使用"）。
    整个项目只有这一处碰内部结构 —— 将来换成别的向量库，也只需要改这一个函数。

    对 main.py 公开（不带下划线），是因为关键词检索需要拿到全库片段来建词表。
    """
    return list(db.docstore._dict.values())


def add_file(content: bytes, filename: str) -> dict:
    """把一份 PDF 收进库：解析 → 切块 → 贴标签 → 存起来。

    content 是文件的二进制内容，filename 是原始文件名。
    """
    # 先落到一个临时文件上，因为 PyPDFLoader 需要一个"硬盘上的路径"
    temp_path = f"./temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        # ① 解析：PDF → 一堆带页码的文字
        docs = PyPDFLoader(temp_path).load()

        # ② 切块：长文档切成小块，方便以后精确检索
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        split_docs = splitter.split_documents(docs)

        # 守卫：这份 PDF 一个字都没读出来（可能是扫描件 / 图片版）
        if not split_docs:
            raise KnowledgeError(
                "这份 PDF 里读不到文字，可能是扫描件或图片版，请换一份能选中文字的 PDF"
            )

        # ③ 给每一段贴上「来自哪个文件」的标签，将来好显示引用来源
        #    这里是个 for 循环：把 split_docs 里每一段依次装进 d 这个盒子
        for d in split_docs:
            d.metadata["file_name"] = filename

        # ④ 入库：库已存在 → 先查重再追加；不存在 → 新建一个
        db = load_db()
        if db is None:
            db = FAISS.from_documents(split_docs, EMBEDDING)
        else:
            if _already_exists(db, filename):
                raise KnowledgeError(f"「{filename}」已经在知识库里了，不用重复上传")
            db.add_documents(split_docs)

        db.save_local(config.DB_PATH)
    finally:
        # finally = 不管上面成功还是中途报错，这里都一定会执行
        # 所以要删的临时文件放在这里，绝不会漏掉
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "msg": "文档入库成功",
        "file": filename,
        "chunk_num": len(split_docs),
        "total": db.index.ntotal,
    }


def _already_exists(db, filename: str) -> bool:
    """这个文件名是不是已经入库了（防止重复上传）"""
    for d in all_chunks(db):
        if d.metadata.get("file_name") == filename:
            return True
    return False


def list_files() -> dict:
    """列出库里已经有哪些文件，各自多少段"""
    db = load_db()
    if db is None:
        return {"files": [], "total": 0}

    # 数一数每个文件各有多少段
    counts = {}
    for d in all_chunks(db):
        name = d.metadata.get("file_name", "未知文件")
        counts[name] = counts.get(name, 0) + 1

    files = [{"file": name, "chunk_num": count} for name, count in counts.items()]
    return {"files": files, "total": db.index.ntotal}


def delete_file(filename: str) -> dict:
    """从库里删掉一整份文件。

    FAISS 不能直接删单条向量，所以做法是：把这文件之外的片段全挑出来，
    删掉整个库，再用挑出来的片段重建。
    """
    db = load_db()
    if db is None:
        raise KnowledgeError("知识库还是空的，没有可删除的文件")

    chunks = all_chunks(db)

    # 把「不属于这个文件」的段挑出来，这些是要保留的
    keep = [d for d in chunks if d.metadata.get("file_name") != filename]

    # 保留的数量和原来一样多 → 说明一段都没删掉 → 这个文件根本不在库里
    if len(keep) == len(chunks):
        raise KnowledgeError(f"知识库里没有「{filename}」这个文件")

    shutil.rmtree(config.DB_PATH)
    if keep:
        FAISS.from_documents(keep, EMBEDDING).save_local(config.DB_PATH)

    return {"msg": "已删除", "file": filename, "left": len(keep)}
