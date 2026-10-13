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

import config
from rag import knowledge, retrieval, generation, session, settings


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
    # no-cache 很要紧：这个页面是单文件前端，改了 HTML 之后如果浏览器还捧着
    # 旧缓存跑，新代码就等于没写 —— 实测踩过这个坑（会话编号存 localStorage
    # 那个修复，就是因为缓存没生效而"看起来没用"）。
    return FileResponse("index.html", headers={"Cache-Control": "no-cache"})


# ============================================================
# 知识库：收文件 / 列清单 / 删文件 —— 全部转交档案室
# ============================================================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
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
        raise HTTPException(status_code=400, detail="知识库还是空的，请先上传一份文档")

    # ⓪ 查询改写：把追问补成能独立检索的问题（无历史则原样返回，不会多花钱）
    #    注意【只用于检索】—— 答案仍按用户原话生成，因为模型本来就能看到完整历史，
    #    而改写句丢了原话的细节，拿它当问题反而更容易答偏。
    if config.QUERY_REWRITE:
        search_query = generation.rewrite_query(req.session_id, req.question)
    else:
        search_query = req.question     

    # ①② 检索 + 拼资料（情报组和文案纯函数，用法和改造前一样）
    corpus = knowledge.all_chunks(db)
    results, rerank_used = retrieval.retrieve(db, search_query, corpus)
    context, sources = generation.build_context(results)

    # 提示条：把"用户需要知道、但那个数字表达不了"的事拼成一句话 —— 前端会渲染成黄色提示。
    # 为什么相关度数字里不说这些？见 generation._ratios 的说明：
    # 数字只负责"组内谁更相关"，"整体可不可信"单独说，两者才不会互相打架。
    notes = []
    if not rerank_used:
        notes.append("重排模型未就绪，本次按向量检索结果作答")
    weak = generation.weak_match_note(results)
    if weak:
        notes.append(weak)
    note = "　".join(notes)

    # 这是个【同步生成器】：FastAPI 会一段一段地把它取出来推给浏览器，
    # 而不是等它全部跑完。这就是"流式"的实现方式。
    def event_stream():
        # 先把来源推过去 —— 前端收到就能建好卡片，
        # 后面答案里流出 [n] 时立刻能连上跳转目标
        yield sse("sources", {"sources": sources, "note": note})

        try:
            stats = {}          # 空盒子：交给 ask_stream 把这一轮的用量装进去
            for piece in generation.ask_stream(req.session_id, req.question, context, stats):
                yield sse("token", {"t": piece})
        except Exception as e:
            # 流已经开始，HTTP 状态码改不了，只能把错误当成一条"事件"发过去。
            #
            # 这里特意兜住【所有】异常，而不是只兜 GenerationError：
            # 中途断网 / 超时抛的是普通网络异常，不兜住的话浏览器会看到
            # 答案莫名其妙断在半截，而且完全没有提示。
            yield sse("error", {"detail": f"{type(e).__name__}: {e}"})
            return

        # 流正常结束：把这一轮的「用量」和「记忆占用」一起告诉前端
        yield sse("done", {
            "usage": stats,
            "context": generation.context_status(req.session_id),
        })

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


@app.get("/sessions")
def list_sessions():
    """历史会话清单（最近用过的在前）—— 前端拿它渲染「历史会话」列表"""
    return {"sessions": session.list_sessions()}


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str):
    """拉回某个会话的历史消息。

    为什么需要它：历史虽然存在服务端，但页面上没有 —— 刷新一下聊天区就空了。
    有了这个接口，前端就能把对话"还原"出来，用户才看得见自己聊过什么。
    （记忆只存了问答原文，没存检索到的资料，所以还原出来的答案里 [n]
      只是普通文字、点不动 —— 这是当前取舍。）
    """
    return {"messages": session.get(session_id)}


# ============================================================
# 模型设置：在网页上切换"用哪家大模型"
# ============================================================
class SettingsRequest(BaseModel):
    provider: str = config.CUSTOM_PROVIDER   # 哪一家（预设 id 或 "custom"）
    base_url: str
    model: str
    api_key: str = ""      # 允许留空 = 沿用这家原来那把（本地 Ollama 也本来就不需要）


@app.get("/settings")
def get_settings():
    """当前设置（脱敏）+ 预设清单。

    注意 public_view() 每家只报"密钥有没有填"，
    **密钥真值永远不会出现在这个响应里。**
    """
    return {
        "current": settings.public_view(),
        "presets": config.LLM_PRESETS,
        "custom_id": config.CUSTOM_PROVIDER,
    }


@app.post("/settings")
def save_settings(req: SettingsRequest):
    settings.save(
        req.provider.strip() or config.CUSTOM_PROVIDER,
        req.base_url.strip(),
        req.model.strip(),
        req.api_key.strip(),
    )
    # 换了模型，旧记忆里全是"上一个模型说过的话"，留着只会让它精神分裂
    session.clear_all()
    return {"msg": "设置已保存，会话记忆已清空", "current": settings.public_view()}


@app.post("/settings/test")
def test_settings(req: SettingsRequest):
    """发一句最短的话试试通不通。**只测试，不保存任何东西。**"""
    # 密钥留空 = 用这一家已经存着的那把（网页上不想重新粘贴时走这条路）
    saved = settings.get(req.provider.strip()) or {}
    api_key = req.api_key.strip() or saved.get("api_key", "")

    try:
        llm = generation.make_llm(req.base_url.strip(), req.model.strip(), api_key)
        reply = llm.invoke("只回答两个字：你好").content
        return {"ok": True, "msg": "连接成功，模型回复：" + reply.strip()[:40]}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # 本地地址连不上时，给一句人话 —— 否则只会看到干巴巴的 "502"
        url = req.base_url.strip()
        if "localhost" in url or "127.0.0.1" in url:
            msg += "　←　连不上本地服务：是不是没启动 Ollama？"
        return {"ok": False, "msg": msg}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
