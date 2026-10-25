from functools import lru_cache

import jieba
from langchain_core.documents import Document as TextDocument
from rank_bm25 import BM25Okapi

from app.core.models import embedding_profile, get_embeddings
from app.db.session import SessionLocal
from app.repositories.documents import active_chunks
from app.retrieval import index
from app.retrieval.reranker import is_available, rerank_documents
from app.services.documents import require_course


def tokenize(text: str):
    return [word.lower() for word in jieba.cut_for_search(text) if word.strip()]


@lru_cache(maxsize=16)
def keyword_index(course_id: str, versions: tuple[str, ...]):
    with SessionLocal() as session:
        rows = active_chunks(session, course_id)
        rows = [row for row in rows if row[2].id in versions]
        ids = [row[0].id for row in rows]
        tokens = [tokenize(row[0].text) or ["__empty__"] for row in rows]
    return ids, tokens, BM25Okapi(tokens) if tokens else None


def search(course_id: str, query: str, top_k=4, semantic=True, rerank=False):
    with SessionLocal() as session:
        require_course(session, course_id)
        rows = active_chunks(session, course_id)
        records = {chunk.id: {
            "id": chunk.id, "document_id": document.id, "version_id": version.id,
            "version": version.number, "source": version.filename, "text": chunk.text,
            "page": chunk.page, "start_index": chunk.start_index,
        } for chunk, document, version in rows}
        versions = tuple(sorted({version.id for _, _, version in rows}))
        profiles = {version.embedding_profile for _, _, version in rows}
    trace = {"semantic": False, "keyword": False, "reranked": False, "candidates": 0, "note": ""}
    if not records:
        return {"results": [], "trace": trace}
    ids, tokens, bm25 = keyword_index(course_id, versions)
    query_tokens = tokenize(query)
    lexical = []
    if bm25:
        scores = bm25.get_scores(query_tokens)
        matching = [position for position, words in enumerate(tokens) if set(words).intersection(query_tokens)]
        lexical = [ids[position] for position in sorted(matching, key=lambda position: scores[position], reverse=True)[:20]]
        trace["keyword"] = True
    vector = []
    if semantic:
        try:
            profile = embedding_profile()
            if profiles != {profile}:
                trace["note"] = "嵌入模型配置已变化，请逐份替换或重新处理资料；暂用关键词搜索。"
            else:
                result = index.collection(course_id, profile).query(
                    query_embeddings=[get_embeddings().embed_query(query)],
                    n_results=min(20, len(records)), where={"version_id": {"$in": list(versions)}},
                    include=["distances"],
                )
                vector = [chunk_id for chunk_id in result["ids"][0] if chunk_id in records]
                trace["semantic"] = True
        except Exception:
            trace["note"] = "语义搜索暂不可用，已使用关键词搜索。请检查嵌入模型配置与网络。"
    fusion = {}
    for ranking in (vector, lexical):
        for rank, chunk_id in enumerate(ranking, 1):
            fusion[chunk_id] = fusion.get(chunk_id, 0) + 1 / (60 + rank)
    ranked = [records[chunk_id] for chunk_id in sorted(fusion, key=fusion.get, reverse=True)]
    trace["candidates"] = len(ranked)
    if rerank and ranked:
        if is_available():
            docs, error = rerank_documents(query, [TextDocument(page_content=row["text"], metadata=row.copy()) for row in ranked], top_k)
            if error:
                trace["note"] = "重排暂不可用，保留混合检索顺序。"
            else:
                trace["reranked"] = True
                ranked = [doc.metadata for doc in docs]
        else:
            trace["note"] = "未配置可用的重排服务，保留召回顺序。"
    # A concurrent delete or replacement must not leak retired content.
    with SessionLocal() as session:
        require_course(session, course_id)
        allowed = {chunk.id for chunk, _, _ in active_chunks(session, course_id)}
    return {"results": [row for row in ranked if row["id"] in allowed][:top_k], "trace": trace}
