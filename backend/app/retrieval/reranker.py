"""重排（Rerank）：用交叉编码器对粗召回结果做精排。

为什么需要重排：
    向量检索（双编码器）把问题和文档分别编码成向量再算距离，速度快但精度有限。
    重排模型（交叉编码器）把「问题 + 文档」拼在一起送进模型打分，
    能捕捉两者的细粒度交互，精度显著更高，但慢得多。
    所以工业界标准做法是两阶段：先用向量/BM25 快速粗召回几十条，再用重排精排取前几条。

接口说明：
    硅基流动、Jina、Cohere 都采用同一套重排协议（自建 vLLM 服务的 /v1/rerank 也兼容），
    请求体为 {model, query, documents, top_n}，返回 {results: [{index, relevance_score}]}，
    因此这里用 urllib 直接调用即可，无需为每家写适配器。
"""

import json
import math
import os
import urllib.error
import urllib.request

from app.core.models import (
    RERANK_DISABLED,
    RERANK_MODEL,
    RERANK_PROVIDER,
    RERANK_PROVIDERS,
)

# 单次重排请求最多送多少条候选（服务商普遍限制 1000 条以内）
_MAX_CANDIDATES = 100


def status(rerank_on: bool = None) -> dict:
    """返回重排功能的可用状态，供界面展示。

    Args:
        rerank_on: 界面开关状态，为 None 时只检查服务商配置。

    Returns:
        包含启用状态、服务商、模型名和不可用原因的字典。
    """
    if RERANK_PROVIDER == RERANK_DISABLED:
        return {
            "enabled": False,
            "provider": RERANK_DISABLED,
            "model": "",
            "reason": "配置中已关闭重排（RERANK_PROVIDER=none）",
        }

    conf = RERANK_PROVIDERS.get(RERANK_PROVIDER)
    if conf is None:
        available = "、".join(list(RERANK_PROVIDERS) + [RERANK_DISABLED])
        return {
            "enabled": False,
            "provider": RERANK_PROVIDER,
            "model": "",
            "reason": f"未知的重排服务商，可选：{available}",
        }

    model = RERANK_MODEL or conf["default_model"]
    if conf["key_env"] and (
        not os.getenv(conf["key_env"], "").strip()
        or "你的" in os.getenv(conf["key_env"], "")
    ):
        return {
            "enabled": False,
            "provider": RERANK_PROVIDER,
            "model": model,
            "reason": f"未配置 {conf['key_env']}，已自动跳过重排",
        }

    return {
        "enabled": bool(rerank_on) if rerank_on is not None else True,
        "provider": RERANK_PROVIDER,
        "model": model,
        "reason": "" if rerank_on is None or rerank_on else "界面开关已关闭重排",
    }


def is_available() -> bool:
    """检查重排服务配置是否可用。

    Returns:
        服务商及其鉴权配置可用时为 True。
    """
    return status()["enabled"]


def _call_api(query: str, documents: list[str], top_n: int) -> list[dict]:
    """调用重排接口，返回按相关度降序的 results 列表。

    Args:
        query: 用于检索的查询文本。
        documents: 待重排的原文列表，只发送前 100 条。
        top_n: 期望返回的候选条数。

    Returns:
        服务商返回的候选位置与相关度分数记录。
    """
    conf = RERANK_PROVIDERS[RERANK_PROVIDER]
    url = conf["base_url"].rstrip("/") + "/rerank"
    payload = {
        "model": RERANK_MODEL or conf["default_model"],
        "query": query,
        "documents": documents[:_MAX_CANDIDATES],
        "top_n": top_n,
        "return_documents": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.getenv(conf['key_env'], '')}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8")).get("results", [])


def rerank_documents(query: str, docs: list, top_n: int) -> tuple[list, str]:
    """按服务商的相关度结果重新排列候选片段。

    成功时将分数写入片段元数据，失败时保留召回顺序。

    Args:
        query: 用于检索的查询文本。
        docs: 待重排的 LangChain 文本片段。
        top_n: 最多保留的片段数。

    Returns:
        选中的片段列表与错误说明。成功或没有候选时，说明为空。
    """
    if not docs:
        return docs, ""

    try:
        results = _call_api(query, [d.page_content for d in docs], top_n)
    except urllib.error.HTTPError as e:
        return docs[:top_n], f"重排接口返回 {e.code}"
    except Exception as e:
        return docs[:top_n], f"重排失败（{type(e).__name__}）"

    reranked = []
    seen = set()
    if not isinstance(results, list):
        return docs[:top_n], "重排结果格式无效"
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score")
        if (
            type(index) is not int
            or not 0 <= index < min(len(docs), _MAX_CANDIDATES)
            or index in seen
        ):
            continue
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        seen.add(index)
        doc = docs[index]
        doc.metadata["rerank_score"] = score
        reranked.append(doc)

    # 接口异常返回空结果时退回原顺序，避免检索结果丢失
    return (reranked[:top_n], "") if reranked else (docs[:top_n], "重排未返回有效结果")
