"""
main.py —— 前台

职责只有一个：接客。
  收到上传  → 转交档案室（knowledge）
  收到提问  → 依次转交档案室、情报组、文案，然后把结果拼起来返回
  管理知识库 → 转交档案室

所有业务逻辑都在 rag/ 里。这个文件应该越薄越好 ——
薄到"想看业务怎么实现的，根本不用打开这个文件"。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from rag import knowledge, retrieval, generation


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时先把重排模型准备好（首次要下载约 1GB）。
    # 提前做掉，免得用户第一次提问时盯着"思考中…"以为是卡死了。
    retrieval.warmup()
    yield


app = FastAPI(title="RAG文档问答服务", lifespan=lifespan)


# ============================================================
# 错误翻译层
# ============================================================
# rag 层只负责说"出什么事了"（抛 KnowledgeError / GenerationError），
# 至于该回给浏览器 400 还是 502，是网页这一层的事 —— 翻译工作统一放在这里。
@app.exception_handler(knowledge.KnowledgeError)
@app.exception_handler(generation.GenerationError)
def handle_business_error(request, exc):
    # 知识库的问题（文件重复、读不出文字）→ 400，是"你的请求有问题"
    # 大模型的问题（重试仍失败）      → 502，是"我依赖的下游服务有问题"
    status = 502 if isinstance(exc, generation.GenerationError) else 400
    return JSONResponse(status_code=status, content={"detail": str(exc)})


# ============================================================
# 网页
# ============================================================
@app.get("/")
def index():
    return FileResponse("index.html")


# ============================================================
# 知识库：收文件 / 列清单 / 删文件 —— 全部转交档案室
# ============================================================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()          # "把上传的内容读出来"是网页这一层的事
    return knowledge.add_file(content, file.filename)


@app.get("/documents")
def list_documents():
    return knowledge.list_files()


@app.delete("/documents")
def delete_document(file: str):
    return knowledge.delete_file(file)


# ============================================================
# 提问：把三个模块串成一条线
# ============================================================
# 这里用同步的 def 而不是 async def：因为里面全是"阻塞式"的调用
# （读库、jieba 分词、重排、调大模型）。FastAPI 会把同步接口丢到线程池里跑，
# 不会卡住其它请求；写成 async 反而会把整个服务堵住。
@app.get("/query")
def query(question: str):
    # ① 档案室：取出向量库和全库片段
    db = knowledge.load_db()
    if db is None:
        raise HTTPException(status_code=400, detail="知识库还是空的，请先上传一份 PDF")
    corpus = knowledge.all_chunks(db)

    # ② 情报组：两路召回 + 融合 + 重排 → [(片段, 分数)]，最相关在前
    results, rerank_used = retrieval.retrieve(db, question, corpus)

    # ③ 文案：拼提示词，再交给大模型写答案
    prompt, sources = generation.make_prompt(question, results)
    answer = generation.ask(prompt)

    note = "" if rerank_used else "重排模型未就绪，本次按向量检索结果作答"
    return {"answer": answer, "sources": sources, "note": note}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
