"""
main.py —— 前台

职责只有一个：接客。
  收到上传  → 转交档案室（knowledge）
  收到提问  → 依次转交档案室、情报组、文案，把结果包成 SSE 流推回去
  管理知识库 → 转交档案室

所有业务逻辑都在 rag/ 里。这个文件应该越薄越好 ——
薄到"想看业务怎么实现的，根本不用打开这个文件"。
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from rag import knowledge, retrieval, generation, session


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
# rag 层只负责说"出什么事了"，至于该回给浏览器什么状态码，是网页这一层的事。
#
# 注意：GenerationError（大模型挂了）【不】在这里处理。
# 因为它必然发生在"流已经开始之后"，那时 HTTP 状态码早就发出去了、改不了，
# 只能在流里面发一条 event: error —— 见下面 chat() 里那段 try/except。
@app.exception_handler(knowledge.KnowledgeError)
def handle_knowledge_error(request, exc):
    # 知识库的问题（文件重复、读不出文字）→ 400，是"你的请求有问题"
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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
# 提问：把三个模块串起来，边生成边推给浏览器
# ============================================================
class ChatRequest(BaseModel):
    question: str       # 用户这一轮的问题
    session_id: str     # 会话编号，前端生成并一直带着


class SessionRequest(BaseModel):
    session_id: str


def sse(event: str, data: dict) -> str:
    """把一段数据包成 SSE 的格式：event 行 + data 行 + 空行。

    那个空行是【分隔符】—— 少了它，两段数据会被浏览器当成一段。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    # 守卫：知识库为空 —— 此刻"还没开始流"，所以还能正常返回 400
    db = knowledge.load_db()
    if db is None:
        raise HTTPException(status_code=400, detail="知识库还是空的，请先上传一份 PDF")

    # ①② 检索 + 拼资料（情报组和文案纯函数，用法和改造前一样）
    corpus = knowledge.all_chunks(db)
    results, rerank_used = retrieval.retrieve(db, req.question, corpus)
    context, sources = generation.build_context(results)
    note = "" if rerank_used else "重排模型未就绪，本次按向量检索结果作答"

    # 这是个【同步生成器】：FastAPI 会一段一段地把它取出来推给浏览器，
    # 而不是等它全部跑完。这就是"流式"的实现方式。
    def event_stream():
        # 先把来源推过去 —— 前端收到就能建好卡片，
        # 后面答案里流出 [n] 时立刻能连上跳转目标
        yield sse("sources", {"sources": sources, "note": note})

        try:
            for piece in generation.ask_stream(req.session_id, req.question, context):
                yield sse("token", {"t": piece})
        except Exception as e:
            # 流已经开始，HTTP 状态码改不了，只能把错误当成一条"事件"发过去。
            #
            # 这里特意兜住【所有】异常，而不是只兜 GenerationError：
            # 中途断网 / 超时抛的是普通网络异常，不兜住的话浏览器会看到
            # 答案莫名其妙断在半截，而且完全没有提示。
            yield sse("error", {"detail": f"{type(e).__name__}: {e}"})
            return

        yield sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # no-cache：别让浏览器或中间的代理把这条流缓存住
        # X-Accel-Buffering: no：叫 nginx 之类的反向代理别攒着，来一点转一点
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/session/reset")
def reset_session(req: SessionRequest):
    session.reset(req.session_id)
    return {"msg": "会话已清空"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
