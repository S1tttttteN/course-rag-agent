"""API contract checks against an isolated on-disk database."""

import os
import sys
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

_temporary = tempfile.TemporaryDirectory()
os.environ["COURSE_AGENT_DATA_DIR"] = _temporary.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
app.state.indexing_enabled = False


class CourseApiTests(unittest.TestCase):
    """验证课程接口的数据持久化、输入校验和访问边界。"""

    def test_persistence_and_stable_id_after_rename(self):
        """验证改名保留课程 ID，重新启动客户端后仍可读取课程。"""
        with TestClient(app) as client:
            created = client.post("/api/courses", json={"name": "  操作系统  ", "description": "进程与内存"})
            self.assertEqual(created.status_code, 201)
            course_id = created.json()["id"]
            self.assertEqual(created.json()["name"], "操作系统")
            renamed = client.patch(f"/api/courses/{course_id}", json={"name": "操作系统复习"})
            self.assertEqual(renamed.status_code, 200)
            self.assertEqual(renamed.json()["id"], course_id)
        with TestClient(app) as client:
            stored = client.get(f"/api/courses/{course_id}")
            self.assertEqual(stored.status_code, 200)
            self.assertEqual(stored.json()["name"], "操作系统复习")
            self.assertTrue(Path(_temporary.name, "app.sqlite3").exists())
            self.assertEqual(client.delete(f"/api/courses/{course_id}").status_code, 204)

    def test_invalid_names_and_updates_do_not_corrupt_catalog(self):
        """验证无效创建和更新请求被拒绝，课程目录保持一致。"""
        with TestClient(app) as client:
            before = client.get("/api/courses").json()
            for name in ["", "   ", "a" * 81, None]:
                self.assertEqual(client.post("/api/courses", json={"name": name}).status_code, 422)
            self.assertEqual(client.get("/api/courses").json(), before)
            created = client.post("/api/courses", json={"name": "课程"}).json()
            for body in [{}, {"name": None}, {"description": None}, {"other": "field"}]:
                self.assertEqual(client.patch(f"/api/courses/{created['id']}", json=body).status_code, 422)
            self.assertEqual(client.get(f"/api/courses/{created['id']}").json()["name"], "课程")
            client.delete(f"/api/courses/{created['id']}")

    def test_delete_one_course_leaves_another_untouched(self):
        """验证课程删除操作仅作用于目标课程。"""
        with TestClient(app) as client:
            first = client.post("/api/courses", json={"name": "同名课程"}).json()
            second = client.post("/api/courses", json={"name": "同名课程"}).json()
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(client.delete(f"/api/courses/{first['id']}").status_code, 204)
            self.assertEqual(client.get(f"/api/courses/{first['id']}").status_code, 404)
            self.assertEqual(client.get(f"/api/courses/{second['id']}").status_code, 200)
            client.delete(f"/api/courses/{second['id']}")

    def test_missing_and_malformed_course_ids(self):
        """验证缺失课程和格式错误的课程 ID 返回对应错误。"""
        with TestClient(app) as client:
            self.assertEqual(client.get(f"/api/courses/{uuid4()}").status_code, 404)
            self.assertEqual(client.get("/api/courses/not-an-id").status_code, 422)

    def test_cors_only_allows_local_frontend(self):
        """验证课程接口只向允许的本地前端返回 CORS 许可。"""
        with TestClient(app) as client:
            headers = {"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"}
            self.assertEqual(client.options("/api/courses", headers=headers).status_code, 200)
            headers["Origin"] = "https://example.com"
            self.assertEqual(client.options("/api/courses", headers=headers).status_code, 400)


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        _temporary.cleanup()
