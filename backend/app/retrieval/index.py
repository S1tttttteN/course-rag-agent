"""管理本地 Chroma 向量集合。哪些资料版本可用于检索由 SQLite 决定。"""

import hashlib
from threading import Lock

from app.core.config import INDEX_DIR

_client = None
_client_lock = Lock()


def client():
    """按需创建并复用本地 Chroma 客户端。

    Returns:
        进程内共享的持久化客户端。
    """
    global _client
    with _client_lock:
        if _client is None:
            import chromadb
            from chromadb.config import Settings

            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(
                str(INDEX_DIR),
                settings=Settings(
                    anonymized_telemetry=False,
                    chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
                ),
            )
    return _client


def close():
    """关闭当前 Chroma 客户端并清除缓存引用。"""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


def _course_prefix(course_id: str):
    """生成课程下所有向量集合共用的前缀。"""
    return f"course_{course_id.replace('-', '')}_"


def _collection_names(prefix: str):
    """枚举具有指定前缀的集合名称。"""
    for item in client().list_collections():
        name = item.name if hasattr(item, "name") else item
        if name.startswith(prefix):
            yield name


def collection_name(course_id: str, profile: str):
    """计算课程和嵌入配置对应的向量集合名。

    Args:
        course_id: 课程 ID。
        profile: 包含嵌入服务商与模型的配置标识。

    Returns:
        以课程 ID 和配置摘要构成的集合名。
    """
    return _course_prefix(course_id) + hashlib.sha256(profile.encode()).hexdigest()[:12]


def collection(course_id: str, profile: str):
    """获取或创建课程和嵌入配置对应的向量集合。

    Args:
        course_id: 课程 ID。
        profile: 包含嵌入服务商与模型的配置标识。

    Returns:
        由调用方提供向量的 Chroma 集合。
    """
    return client().get_or_create_collection(
        collection_name(course_id, profile), embedding_function=None
    )


def remove_document(course_id: str, document_id: str | None):
    """清理指定资料或整门课程的向量。

    Args:
        course_id: 课程 ID。
        document_id: 待删除的资料 ID，为 None 时删除课程的全部向量集合。
    """
    for name in _collection_names(_course_prefix(course_id)):
        if document_id is None:
            client().delete_collection(name)
        else:
            client().get_collection(name, embedding_function=None).delete(
                where={"document_id": document_id}
            )


def remove_retired_versions(course_id: str, document_id: str, active_version_id: str):
    """回收指定资料的旧向量，保留刚发布的版本。"""
    where = {"$and": [
        {"document_id": document_id},
        {"version_id": {"$ne": active_version_id}},
    ]}
    for name in _collection_names(_course_prefix(course_id)):
        client().get_collection(name, embedding_function=None).delete(where=where)


def reconcile(active_versions: set[str]):
    """清理中断写入和退役版本的向量，保留当前可用版本。

    Args:
        active_versions: 应保留向量的可用版本 ID 集合。
    """
    for name in _collection_names("course_"):
        store = client().get_collection(name, embedding_function=None)
        records = store.get(include=["metadatas"])
        stale = [
            chunk_id
            for chunk_id, metadata in zip(records["ids"], records["metadatas"])
            if metadata.get("version_id") not in active_versions
        ]
        for offset in range(0, len(stale), 500):
            store.delete(ids=stale[offset : offset + 500])
