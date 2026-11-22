"""保存本地课程资料快照并校验评测题的证据标注。"""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile


RETRIEVAL_CATEGORIES = ("exact", "paraphrase", "multi_evidence")
QA_CATEGORIES = (*RETRIEVAL_CATEGORIES, "followup", "insufficient")


def write_json(path, value):
    """原子写入本地 JSON，保留上一份完整报告直到新报告就绪。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(value, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
        except Exception:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_json(path):
    """读取评测过程生成的 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(course_name, destination):
    """只读打开当前资料库并复制指定课程的已发布原件。

    Args:
        course_name: 用户资料库中的准确课程名称。
        destination: 尚不存在的冻结快照目录。

    Returns:
        包含资料版本、文件哈希和当前索引配置的清单。
    """
    from app.core.config import DATABASE_PATH
    from app.core.models import CHUNK_OVERLAP, CHUNK_SIZE, embedding_profile
    from app.services.documents import raw_path

    if destination.exists():
        raise ValueError("快照已存在。请使用新目录冻结另一个版本。")
    with closing(sqlite3.connect(f"{DATABASE_PATH.as_uri()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        courses = connection.execute(
            "SELECT id FROM courses WHERE name = ?", (course_name,)
        ).fetchall()
        if len(courses) != 1:
            raise ValueError("课程名称必须匹配唯一一门课程。")
        course_id = courses[0]["id"]
        rows = connection.execute(
            """SELECT d.id AS document_id, v.id AS version_id, v.filename, v.number,
                      v.content_hash, v.chunk_count, v.embedding_profile
               FROM documents d JOIN document_versions v ON v.id = d.active_version_id
               WHERE d.course_id = ? AND v.status = 'ready'
               ORDER BY d.name, d.id""",
            (course_id,),
        ).fetchall()
    if not rows:
        raise ValueError("指定课程没有已发布资料。")

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "course": course_name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_profile": embedding_profile(),
        "documents": [],
    }
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        snapshot = Path(staging) / "snapshot"
        (snapshot / "files").mkdir(parents=True)
        for index, row in enumerate(rows, 1):
            key = f"D{index:02d}"
            filename = row["filename"]
            original = raw_path(course_id, row["document_id"], row["version_id"], filename)
            data = original.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != row["content_hash"]:
                raise ValueError(f"{key} 原件校验失败。请重新冻结资料。")
            copy = f"{key}{Path(filename).suffix.lower()}"
            (snapshot / "files" / copy).write_bytes(data)
            manifest["documents"].append({
                "key": key,
                "filename": filename,
                "file": copy,
                "sha256": digest,
                "version": row["number"],
                "version_id": row["version_id"],
                "original_chunks": row["chunk_count"],
                "original_embedding_profile": row["embedding_profile"],
            })
        write_json(snapshot / "manifest.json", manifest)
        snapshot.rename(destination)
    return manifest


def snapshot_file(snapshot, document):
    """校验冻结文件的路径和哈希，返回可供解析及索引使用的原件。"""
    filename = document["file"]
    if Path(filename).name != filename or Path(document["filename"]).name != document["filename"]:
        raise ValueError("快照文件名不合法。")
    folder = (snapshot / "files").resolve()
    path = (folder / filename).resolve()
    if path.parent != folder or hashlib.sha256(path.read_bytes()).hexdigest() != document["sha256"]:
        raise ValueError(f"{document['key']} 快照原件校验失败。")
    return path


def normalized(text):
    """压平分块器保留的换行和设计文档空格。"""
    return " ".join(text.split())


def resolve_evidence(groups, chunks):
    """把每组必需证据映射到所有可互换的片段 ID。

    Args:
        groups: 按事实组织的证据组，每组包含一个或多个来源锚点。
        chunks: 以快照资料代号为键的片段，包含 id、text 和 page。

    Returns:
        按组排列的命中片段 ID 集合。

    Raises:
        ValueError: 标注为空或有锚点无法定位。
    """
    resolved = []
    for alternatives in groups:
        if not alternatives:
            raise ValueError("必需证据组至少需要一个原文定位。")
        ids = set()
        for source in alternatives:
            anchor = normalized(source["anchor"])
            matches = {
                chunk["id"] for chunk in chunks.get(source["document"], [])
                if anchor and anchor in normalized(chunk["text"])
                and ("page" not in source or chunk["page"] == source["page"])
            }
            if not matches:
                raise ValueError(f"{source['document']} 的原文定位未匹配：{anchor}")
            ids.update(matches)
        resolved.append(ids)
    return resolved


def load_cases(path):
    """校验题库结构，并将复用题展开为统一的多轮问答用例。"""
    data = read_json(path)
    retrieval = data["retrieval"]
    references = {item["id"]: item for item in retrieval}
    if len(references) != len(retrieval):
        raise ValueError("检索题 ID 重复。")
    for item in retrieval:
        if item["category"] not in RETRIEVAL_CATEGORIES or not item["evidence"] or not item["reference"]:
            raise ValueError(f"检索题缺参考要点或证据：{item['id']}")
    qa = []
    for item in data["qa"]:
        if "from_retrieval" in item:
            reference = references[item["from_retrieval"]]
            turn = {key: reference[key] for key in ("question", "reference", "evidence")}
            item = {"id": item["id"], "category": reference["category"], "turns": [turn]}
        if item["category"] not in QA_CATEGORIES or not item["turns"]:
            raise ValueError(f"无效问答场景：{item['id']}")
        for turn in item["turns"]:
            if not turn["question"] or not turn["reference"]:
                raise ValueError(f"问答标注不完整：{item['id']}")
            if item["category"] == "insufficient":
                if not turn.get("gap") or turn["evidence"]:
                    raise ValueError(f"资料不足题须标明缺口：{item['id']}")
            elif not turn["evidence"]:
                raise ValueError(f"问答题缺少证据：{item['id']}")
        qa.append(item)
    if len({item["id"] for item in qa}) != len(qa):
        raise ValueError("问答场景 ID 重复。")
    return retrieval, qa
