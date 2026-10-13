"""
run_eval.py —— 检索质量评估（纯本地，不调用大模型，零 API 花费）

要回答的问题：**混合检索 + 重排这一轮升级，到底有没有让"找资料"变准？**

做法：把同一批问题分别喂给三种检索配置，看谁把"答案所在的那一段"排得更靠前：

    ① vector        —— 只用向量检索（升级【前】的老做法）
    ② hybrid        —— 向量 + 关键词，RRF 融合（升级【后】、但关掉重排）
    ③ hybrid+rerank —— 再加交叉编码器重排（升级【后】的完整做法＝线上真实配置）

三个指标（都是"越大越好"）：
    Hit@1      第一名就是对的吗 —— 最直观：能不能"一枪命中"
    Recall@5   正确答案有没有进前 5 —— 会不会漏掉
    MRR        正确答案排名的倒数平均值 —— 综合"排得够不够靠前"

⚠️ 一个很要紧的前提：**题集和文档必须有"干扰项"才有意义**。
   如果全文只有 5 段、而每次取前 5 段，那 Recall@5 永远是 100%，测了等于没测。
   所以测试文档刻意做成 6 章 31 节，而且**故意放了很多"长得像、数字不同"的兄弟条款**
   （好几个地方都出现 3 天、5000 元、10 个工作日……）——
   这才是真实文档的样子，也才逼得出"光靠意思像不像"到底够不够。

为什么不用 RAGAs？
    它是用大模型给"答案"打分（忠实度、相关性），每道题要几十次 LLM 调用 ——
    慢、要花钱、还容易被免费模型的限流打回来。
    而"检索准不准"这件事，只要有一份**标注好出处**的题集，就能纯本地算出来，
    既快又是确定的（没有模型随机性）。先把最核心的问题用最便宜的办法量清楚。

为什么复用 rag/ 里的组件、不另写一套检索？
    因为要量的是【你的系统】，不是"我另写的另一套检索"。
    所以这里直接调 VectorRetriever / KeywordRetriever / fuse / 重排 —— 和线上同一批零件。

用法：
    python eval/run_eval.py            # 跑全部，打一张对照表
    python eval/run_eval.py -v         # 再逐题列出每条的命中情况
"""

import os
import sys
import tempfile

# 让脚本能从任意目录跑：把项目根目录塞进 import 搜索路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config                                                    # noqa: E402
from rag import knowledge, retrieval                             # noqa: E402


# ============================================================
# 一、测试文档 —— 六章 31 节，刻意布满"像但不相同"的干扰条款
# ============================================================
# 每节都写得很短（远小于 CHUNK_SIZE=500），好让「一节 = 一块」，
# 这样 location 就能唯一地标记"答案在哪一段"，评估才有干净的靶子。

DOC_PARAGRAPHS = [
    ("heading", "▎ 出差申请"),
    ("p", "员工出差前须填写《出差申请单》，经直属主管与部门负责人双签后，交行政部备案。"),
    ("heading", "▎ 擅自出差"),
    ("p", "未办理申请手续擅自出差的，相关费用由个人自行承担。"),
    ("heading", "▎ 住宿标准"),
    ("p", "住宿费上限：一线城市每晚上限 600 元，二线城市每晚上限 400 元，其他城市每晚上限 300 元。"),
    ("heading", "▎ 交通标准"),
    ("p", "市内交通费每人每天不超过 100 元；跨城出行原则上乘坐二等座。"),
    ("heading", "▎ 伙食补助"),
    ("p", "伙食补助每人每天 80 元，按实际出差天数计发。"),
    ("heading", "▎ 报销时限"),
    ("p", "出差结束后 10 个工作日内提交报销单，并附发票原件与出差申请单复印件。"),
    ("heading", "▎ 延期审批"),
    ("p", "因公延期超过 3 天的，须报分管副总审批后方可继续计算差旅补助。"),
    ("heading", "▎ 绕道自理"),
    ("p", "出差期间因私绕道的，绕道段的交通与住宿费用一律自理。"),
    ("heading", "▎ 制度编号"),
    ("p", "本《差旅报销制度》的归档编号为 HT-2024-0917，行文引用时请标注该编号。"),
    ("heading", "▎ 合同编号"),
    ("p", "采购框架协议的统一编号为 CG-2025-0088，签约时须在合同首页标注该编号。"),
    ("heading", "▎ 门禁设备编号"),
    ("p", "园区门禁设备的报修编号为 SB-2023-0451，报修时请提供该编号。"),
    ("heading", "▎ 培训课程编号"),
    ("p", "新员工必修课《信息安全》的课程编号为 KC-2022-0772。"),

    ("heading", "▎ 会议费标准"),
    ("p", "会议费按每人每天 200 元标准执行，含场地与茶歇。"),
    ("heading", "▎ 会议室预订"),
    ("p", "会议室需提前 3 个工作日预订，由行政部统一安排。"),
    ("heading", "▎ 会议纪要"),
    ("p", "会议结束后 2 个工作日内发出会议纪要，抄送全体参会人。"),
    ("heading", "▎ 外部会议"),
    ("p", "参加外部会议需提前报备，差旅费用按出差标准执行。"),
    ("heading", "▎ 会议制度编号"),
    ("p", "《会议管理办法》的归档编号为 HY-2024-0331，与差旅制度配套执行。"),

    ("heading", "▎ 采购申请"),
    ("p", "单笔金额超过 5000 元的采购，需提交采购申请单并经分管副总审批。"),
    ("heading", "▎ 供应商准入"),
    ("p", "供应商须提供营业执照，近三年无重大违约记录，且年营业额不低于 3000 万元。"),
    ("heading", "▎ 比价要求"),
    ("p", "单笔金额超过 10000 元的采购，须至少取得三家供应商报价进行比价。"),
    ("heading", "▎ 验收流程"),
    ("p", "到货后 5 个工作日内完成验收，验收单需采购部门与使用部门双签。"),
    ("heading", "▎ 付款周期"),
    ("p", "验收合格后 30 个工作日内付款，供应商账期统一为月结。"),

    ("heading", "▎ 票据要求"),
    ("p", "所有发票抬头必须为公司全称，不接受任何个人抬头的发票。"),
    ("heading", "4.2 电子发票"),
    ("p", "电子发票需同时提供行程单或消费明细，单独一张发票不予受理。"),
    ("heading", "▎ 审批权限"),
    ("p", "单笔报销 5000 元以内由部门负责人审批，超过 5000 元的报分管副总审批。"),
    ("heading", "▎ 不予报销"),
    ("p", "餐饮发票无明细、发票抬头不符、超过时限提交的，一律不予报销。"),
    ("heading", "▎ 借款冲销"),
    ("p", "出差借款须在返回后 15 个工作日内冲销，未冲销者暂停新增借款。"),

    ("heading", "▎ 打卡规则"),
    ("p", "每日需打卡两次，上班 9:00 前、下班 18:00 后各一次。"),
    ("heading", "▎ 迟到处理"),
    ("p", "月累计迟到超过 3 次的，扣发当月全勤奖。"),
    ("heading", "▎ 请假审批"),
    ("p", "请假 1 天以内由主管审批，1 天以上报部门负责人，3 天以上报分管副总。"),
    ("heading", "▎ 加班申请"),
    ("p", "加班需提前申请，事后补申请的须在 2 个工作日内提交。"),
    ("heading", "▎ 调休折算"),
    ("p", "加班时长可按 1:1 折算调休，调休须在三个月内使用完毕。"),

    ("heading", "▎ 设备管理"),
    ("p", "办公电脑不得安装未经审批的软件，外接存储设备需到 IT 部门登记。"),
    ("heading", "▎ 数据分级"),
    ("p", "数据分为公开、内部、机密三级，机密数据一律不得外发。"),
    ("heading", "▎ 账号安全"),
    ("p", "账号密码每 90 天更换一次，不得与同事共用账号。"),
    ("heading", "▎ 违规处理"),
    ("p", "违反信息安全规定的，视情节轻重给予警告直至解除劳动合同。"),
]

DOC_TABLE = [
    ["费用类别", "上限"],
    ["住宿（一线）", "600 元/晚"],
    ["市内交通", "100 元/天"],
    ["伙食补助", "80 元/天"],
]


# ============================================================
# 一·五、批量生成的"干扰条款" —— 语料必须先够大，指标才有区分度
# ============================================================
# 为什么非要有它们？
#   上一版语料只有 33 块，而向量粗召回一次就取 20 条 —— 等于把六成语料都捞出来了，
#   "找得准"根本无从谈起，关键词那一路自然显不出任何优势。
#   放进足够多的干扰条款后，前 20 条才是一次真正的挑选，两路召回才能公平比较。
#
# 但干扰条款不能只是凑数，它们必须【语义上很像、字面上不同】才有意义：
#   都是"住宿费上限"的条款，只有金额和适用人群不同 —— 这正是真实文档的样子，
#   也正好卡在纯向量检索的软肋上（它分不清这些条款之间那点数字差别）。
#
# 全部由下标算式生成、不用随机数 —— 跑几遍结果都一样（可复现）。
def build_distractors() -> list:
    out = []

    # ① 住宿 / 交通类补充规定：都带"住宿费""上限"，但人群和金额都不同。
    #    金额刻意避开正确答案里的 600，交通费避开 100 —— 免得干扰项反而"蒙对"。
    scopes = ["外派", "实习生", "兼职", "劳务派遣", "外部顾问", "分公司", "海外派驻"]
    for i, scope in enumerate(scopes):
        for j in range(3):
            hotel = 420 + i * 7 + j * 11          # 最大 484，始终不等于 600
            traffic = 60 + j * 5                  # 60 / 65 / 70，不等于 100
            out.append(("heading", f"▎ {scope}差旅补充规定 {j + 1}"))
            out.append(("p", f"{scope}人员住宿费上限为 {hotel} 元/晚，"
                             f"市内交通费按 {traffic} 元/天执行，超出部分自理。"))

    # ② 归档编号类：按"编号家族"各生成 10 条，长得几乎一样。
    #    问「HY-2024-0331 对应哪份文件」时，纯向量很难从这一堆里挑对，
    #    而关键词检索对 "0331" 这串字符是精确命中的 —— 这正是要量出来的东西。
    #    刻意让干扰编号与正确答案【同前缀】（HT/CG/SB/KC-20xx），才够难。
    for family in ["HT-2024", "CG-2025", "SB-2023", "KC-2022"]:
        for i in range(10):
            code = f"{family}-{100 + i:04d}"       # 0100…0109，不会撞上答案里的 0917/0331/0088/0451/0772
            out.append(("heading", f"▎ {family} 归档编号 {i + 1}"))
            out.append(("p", f"归档编号 {code} 对应《内部制度附件第 {i + 1} 号》，"
                             f"仅供存档登记时核对使用。"))

    # ③ 通用附则：纯凑体量，让"前 20 条"变成真正的筛选。
    for i in range(80):
        out.append(("heading", f"▎ 附则 {i + 1}"))
        out.append(("p", f"本规定第 {i + 1} 条自发布之日起施行，"
                         f"未尽事宜由行政部负责解释与修订。"))

    return out


# ============================================================
# 二、考题集 —— 每题的 expect 是"这段内容应该落在哪个 location 里"
# ============================================================
# kind 只用来分组统计，好看出"关键词检索到底在哪些题型上帮了忙"：
#   exact    —— 靠精确字样（数字）
#   code     —— 专有编号 / 型号（纯向量最典型的软肋）
#   semantic —— 要靠意思理解，提问用词和原文并不相同
CASES = [
    # —— 精确词：原文里有明确的数字/编号 ——
    {"q": "一线城市住宿费每晚上限是多少？",        "expect": "▎ 住宿标准", "kind": "exact"},
    {"q": "市内交通费每人每天上限是多少？",        "expect": "▎ 交通标准", "kind": "exact"},
    {"q": "伙食补助每人每天多少钱？",              "expect": "▎ 伙食补助", "kind": "exact"},
    {"q": "出差报销单要在几个工作日内提交？",      "expect": "▎ 报销时限", "kind": "exact"},
    {"q": "供应商的年营业额门槛是多少？",          "expect": "▎ 供应商准入", "kind": "exact"},
    {"q": "单笔采购超过多少钱需要三家比价？",      "expect": "▎ 比价要求", "kind": "exact"},
    {"q": "验收合格后多少个工作日内付款？",        "expect": "▎ 付款周期", "kind": "exact"},
    {"q": "账号密码多久需要更换一次？",            "expect": "▎ 账号安全", "kind": "exact"},
    {"q": "会议费每人每天的标准是多少？",          "expect": "▎ 会议费标准", "kind": "exact"},
    {"q": "月累计迟到几次会扣全勤奖？",            "expect": "▎ 迟到处理", "kind": "exact"},

    # —— 语义理解：问法和原文用词不同，得靠"意思像不像" ——
    {"q": "出差回来之后最晚什么时候把票据交上去？", "expect": "▎ 报销时限", "kind": "semantic"},
    {"q": "没办手续就自己跑去出差，钱谁掏？",       "expect": "▎ 擅自出差", "kind": "semantic"},
    {"q": "出差时间被拉长了，要找谁批准？",         "expect": "▎ 延期审批", "kind": "semantic"},
    {"q": "加了班想换成休息时间，有什么限制？",     "expect": "▎ 调休折算", "kind": "semantic"},
    {"q": "公司对个人抬头的发票是什么态度？",       "expect": "▎ 票据要求", "kind": "semantic"},
    {"q": "货送到以后要做完哪些事才算数？",         "expect": "▎ 验收流程", "kind": "semantic"},

    # —— 编号类：专有编号是纯向量检索最典型的软肋 ——
    #   干扰语料里塞了 40 条"同前缀、只差四个数字"的编号条款，靠"意思像不像"很难分开；
    #   但关键词检索对这几个字符是精确命中的。这正是当初"型号/条款编号答不上来"的场景。
    {"q": "HT-2024-0917 是哪个制度的编号？",       "expect": "▎ 制度编号", "kind": "code"},
    {"q": "HY-2024-0331 对应哪份文件？",           "expect": "▎ 会议制度编号", "kind": "code"},
    {"q": "CG-2025-0088 是什么编号？",             "expect": "▎ 合同编号", "kind": "code"},
    {"q": "SB-2023-0451 这个编号是干什么用的？",   "expect": "▎ 门禁设备编号", "kind": "code"},
    {"q": "KC-2022-0772 是哪门课的编号？",         "expect": "▎ 培训课程编号", "kind": "code"},
]


# ============================================================
# 三、把测试文档写成 .docx —— 顺便把"结构感知切块"也一起量进去
# ============================================================
def build_test_docx() -> str:
    """生成测试用的 Word 文档，返回它的路径（放在临时目录，不落进项目）。

    为什么绕一圈写成 .docx，不直接用纯文本？
      因为线上真实入口就是"上传文档 → 档案室解析"。直接吃 txt 会绕开
      Word 解析器（手工标题识别、表格提取），量出来的就不是真实链路了。
    """
    from docx import Document

    doc = Document()
    for kind, text in DOC_PARAGRAPHS + build_distractors():
        doc.add_paragraph(text)

    table = doc.add_table(rows=len(DOC_TABLE), cols=len(DOC_TABLE[0]))
    for r, row in enumerate(DOC_TABLE):
        for c, val in enumerate(row):
            table.cell(r, c).text = val

    path = os.path.join(tempfile.mkdtemp(), "差旅报销制度.docx")
    doc.save(path)
    return path


# ============================================================
# 四、三种检索配置 —— 全部复用 rag/retrieval.py 里的真实零件
# ============================================================
def _vector_only(db, question, corpus):
    """① 升级前的做法：只看向量"""
    return retrieval.VectorRetriever(db).search(question, config.RERANK_TOP)


def _hybrid_no_rerank(db, question, corpus):
    """② 加了关键词 + RRF，但关掉重排"""
    vec = retrieval.VectorRetriever(db).search(question, config.VECTOR_K)
    kw = retrieval.KeywordRetriever(corpus).search(question, config.KEYWORD_K)
    fused = retrieval.fuse([vec, kw], config.RRF_WEIGHTS, config.RRF_C)
    return fused[:config.RERANK_TOP]


def _hybrid_rerank(db, question, corpus):
    """③ 线上真实配置：混合检索 + 重排（直接调 retrieve，连降级逻辑都一致）"""
    results, _ = retrieval.retrieve(db, question, corpus)
    return [doc for doc, _ in results]


CONFIGS = [
    ("① 只用向量（升级前）", _vector_only),
    ("② 混合检索（+关键词 RRF）", _hybrid_no_rerank),
    ("③ 混合 + 重排（线上现在）", _hybrid_rerank),
]


# ============================================================
# 五、跑分
# ============================================================
def hit_rank(docs, expect_location) -> int:
    """正确答案被排在第几名？没进前 5 就返回 0。

    判断依据是 location —— 也就是来源卡片上那句"第 3 页 / 一、概述 > 1.2 安装"。
    用它当靶子，是因为它由档案室填好、和用户实际看到的位置说明完全一致。
    """
    for i, d in enumerate(docs, start=1):
        if d.metadata.get("location") == expect_location:
            return i
    return 0


def score_config(db, corpus, name, fn, verbose=False):
    """跑完一批题，算 Hit@1 / Recall@5 / MRR，并按题型分组。"""
    ranks = []
    by_kind = {}

    for case in CASES:
        docs = fn(db, case["q"], corpus)
        r = hit_rank(docs, case["expect"])
        ranks.append(r)
        by_kind.setdefault(case["kind"], []).append(r)

        if verbose:
            mark = "命中第1" if r == 1 else ("命中第%d" % r if r else "未命中")
            print(f"     {mark:<8} [{case['kind']:<8}] {case['q']}")

    n = len(ranks)
    result = {
        "name": name,
        "hit1": sum(1 for r in ranks if r == 1) / n,
        "recall5": sum(1 for r in ranks if r) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }
    for kind, rs in by_kind.items():
        result["kind_" + kind] = sum(1 for r in rs if r) / len(rs)
    return result


def attribute(db, corpus, verbose=False):
    """正确段落"死在哪一层"—— 把每个阶段的功劳分清楚。

    为什么要单独做这件事？
      只看最终命中率，没法分清到底是谁的功劳：既可能是关键词检索把正确段落
      捞进了候选池，也可能它本来就一直在池子里、只是重排最后把它排上来的。
      不弄清楚，就容易把重排的功劳记到关键词头上（或者反过来）。
    """
    layers = ["向量粗召回", "关键词粗召回", "融合候选池", "最终前5（线上）"]
    counts = dict.fromkeys(layers, 0)

    for case in CASES:
        exp = case["expect"]

        vec = retrieval.VectorRetriever(db).search(case["q"], config.VECTOR_K)
        kw = retrieval.KeywordRetriever(corpus).search(case["q"], config.KEYWORD_K)
        fused = retrieval.fuse([vec, kw], config.RRF_WEIGHTS, config.RRF_C)
        final = _hybrid_rerank(db, case["q"], corpus)

        hit = lambda docs: any(d.metadata.get("location") == exp for d in docs)   # noqa: E731
        if hit(vec):
            counts["向量粗召回"] += 1
        if hit(kw):
            counts["关键词粗召回"] += 1
        if hit(fused):
            counts["融合候选池"] += 1
        if hit(final):
            counts["最终前5（线上）"] += 1

        if verbose:
            print(f"     {case['expect']:<16} "
                  f"向量{'✓' if hit(vec) else '✗'} "
                  f"关键词{'✓' if hit(kw) else '✗'} "
                  f"融合{'✓' if hit(fused) else '✗'} "
                  f"最终{'✓' if hit(final) else '✗'}")

    return counts, len(CASES), layers


def fmt(x) -> str:
    return f"{x * 100:5.0f}%"


def main():
    verbose = "-v" in sys.argv

    print("=" * 78)
    print("检索质量评估（纯本地，不调用大模型）")
    print("=" * 78)

    # ① 解析测试文档 —— 走的是线上的真实入口
    docx_path = build_test_docx()
    sections = knowledge._sections_docx(docx_path)
    docs = knowledge._to_documents(sections, "差旅报销制度.docx")
    print(f"\n测试文档：{len(sections)} 节 → 切出 {len(docs)} 块")
    print(f"（每次只取前 {config.RERANK_TOP} 块交给模型，语料越大，「找得准」才越有意义）")

    # 靶子必须干净：一个 location 只能对应一块，否则"命中"就没有唯一答案
    seen = {}
    for d in docs:
        seen[d.metadata["location"]] = seen.get(d.metadata["location"], 0) + 1
    dup = {k: v for k, v in seen.items() if v > 1}
    if dup:
        print(f"\n⚠️ 这些 location 对应了多块，作为靶子不唯一：{dup}")

    missing = [c["expect"] for c in CASES if c["expect"] not in seen]
    if missing:
        print(f"\n⚠️ 这些考题的 expect 在文档里根本不存在：{sorted(set(missing))}")
        print("   （考题写错了，下面的分数不可信）")
        return 1

    # ② 建一个内存向量库（不落盘，绝不碰你真实的 faiss_db）
    from langchain_community.vectorstores import FAISS
    db = FAISS.from_documents(docs, knowledge.EMBEDDING)
    corpus = knowledge.all_chunks(db)
    print(f"\n向量库建好（只在内存里，{len(corpus)} 块，没写进你的 faiss_db）")

    # ③ 三种配置各跑一遍
    results = []
    for name, fn in CONFIGS:
        print(f"\n--- {name} ---")
        results.append(score_config(db, corpus, name, fn, verbose))

    # ④ 对照表
    print()
    print("=" * 78)
    print(f"{'配置':<28}{'Hit@1':>8}{'Recall@5':>10}{'MRR':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<28}{fmt(r['hit1']):>8}{fmt(r['recall5']):>10}{fmt(r['mrr']):>8}")

    base, hybrid, final = results
    print("-" * 78)
    print(f"{'相比「只用向量」的提升':<28}"
          f"{(final['hit1'] - base['hit1']) * 100:>+7.0f}%"
          f"{(final['recall5'] - base['recall5']) * 100:>+9.0f}%"
          f"{(final['mrr'] - base['mrr']) * 100:>+7.0f}%")

    # ⑤ 按题型拆开看 —— 这才能说明"关键词检索到底帮在哪"
    print()
    print("按题型拆（Recall@5）：")
    for kind in sorted({c["kind"] for c in CASES}):
        label = {"exact": "精确数字", "code": "专有编号", "semantic": "语义理解"}.get(kind, kind)
        cnt = sum(1 for c in CASES if c["kind"] == kind)
        row = "   " + f"{label}（{cnt} 题）".ljust(26)
        for r in results:
            row += fmt(r.get("kind_" + kind, 0)).rjust(10)
        print(row)
    print()
    print("（列顺序与上表一致：只用向量 / 混合 / 混合+重排）")

    # ⑥ 归因：正确段落是"死在哪一层"的 —— 分清是召回没捞到，还是排序没排上来
    print()
    print("=" * 78)
    print("归因：正确段落有没有被这一层捞到")
    print("=" * 78)
    counts, total, layers = attribute(db, corpus, verbose)
    for name in layers:
        c = counts[name]
        print(f"   {name:<18}{c:>3}/{total}  {fmt(c / total)}")
    print()
    print("怎么读这几行：")
    print("   · 上一层没捞到、下一层捞到了 → 说明【下一层】补上了召回")
    print("   · 全都捞到了、只有最后一行满 → 说明排序是靠【重排】救回来的")
    return 0


if __name__ == "__main__":
    sys.exit(main())
