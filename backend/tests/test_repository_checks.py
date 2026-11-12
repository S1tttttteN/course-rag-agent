"""Self-checks for the supplementary scanner; all tokens are constructed fixtures."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('repository_checks', Path(__file__).resolve().parents[2] / 'scripts/check_repository.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class RepositoryChecksTests(unittest.TestCase):
    """验证仓库敏感文件规则和公开配置模板。"""

    def test_sensitive_paths_and_public_template(self):
        """验证敏感路径识别及公开模板的例外规则。"""
        for name in ('.env', 'backend/.env.local', 'data/app.sqlite3', 'backup/id_ed25519', 'cert/server.key'):
            with self.subTest(name=name):
                self.assertTrue(checks.sensitive_path(name))
        self.assertFalse(checks.sensitive_path('.env.example'))
        self.assertFalse(checks.sensitive_path('backend/app/core/config.py'))

    def test_rules_return_labels_without_secret_values(self):
        """验证扫描结果只返回命中的规则标签。"""
        values = [b'ghp_' + b'A' * 36, b'sk-' + b'B' * 32, b'-----BEGIN ' + b'OPENSSH PRIVATE KEY-----']
        for value in values:
            with self.subTest(length=len(value)):
                result = checks.content_findings(value)
                self.assertTrue(result)
                self.assertNotIn(value.decode(), str(result))
        value = b'nonstandard-fixture-credential'
        self.assertEqual(checks.content_findings(value, [value]), ['local credential value'])
        self.assertEqual(checks.content_findings('sk-你的硅基流动密钥'.encode()), [])
