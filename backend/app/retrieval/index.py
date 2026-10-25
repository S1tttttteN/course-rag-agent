"""Chroma stores vectors; SQLite decides which versions are visible."""
import hashlib
from threading import Lock

from app.core.config import INDEX_DIR

_client = None
_client_lock = Lock()


def client():
    global _client
    with _client_lock:
        if _client is None:
            import chromadb
            from chromadb.config import Settings
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(str(INDEX_DIR), settings=Settings(
                anonymized_telemetry=False, chroma_api_impl='chromadb.api.rust.RustBindingsAPI'))
    return _client


def close():
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


def collection_name(course_id: str, profile: str):
    return f"course_{course_id.replace('-', '')}_{hashlib.sha256(profile.encode()).hexdigest()[:12]}"


def collection(course_id: str, profile: str):
    return client().get_or_create_collection(collection_name(course_id, profile), embedding_function=None)


def remove_document(course_id: str, document_id: str | None):
    prefix = f"course_{course_id.replace('-', '')}_"
    for item in client().list_collections():
        name = item.name if hasattr(item, "name") else item
        if name.startswith(prefix):
            if document_id is None:
                client().delete_collection(name)
            else:
                client().get_collection(name, embedding_function=None).delete(where={"document_id": document_id})


def reconcile(active_versions: set[str]):
    """Remove interrupted writes and retired versions, never published vectors."""
    for item in client().list_collections():
        name = item.name if hasattr(item, "name") else item
        if not name.startswith("course_"):
            continue
        store = client().get_collection(name, embedding_function=None)
        records = store.get(include=["metadatas"])
        stale = [chunk_id for chunk_id, metadata in zip(records["ids"], records["metadatas"]) if metadata.get("version_id") not in active_versions]
        for offset in range(0, len(stale), 500):
            store.delete(ids=stale[offset:offset + 500])
