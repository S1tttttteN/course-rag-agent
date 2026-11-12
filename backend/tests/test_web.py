"""Public-web URL and extraction contracts, with DNS and HTTP fully replaced."""
import socket
import unittest
from unittest.mock import patch

import httpx
import test_courses  # Initializes the isolated test environment before importing app.
from app.retrieval import web


PUBLIC_DNS = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.215.14', 443))]
HTTP_CLIENT = httpx.Client


class WebTests(unittest.TestCase):
    """使用模拟 DNS 和 HTTP 响应验证公开网页工具。"""

    def setUp(self):
        """安装默认返回公开地址的 DNS 测试替身。"""
        self.dns = self.enterContext(patch('app.retrieval.web.socket.getaddrinfo', return_value=PUBLIC_DNS))

    def client(self, handler):
        """安装由测试函数处理请求的模拟 HTTP 客户端。

        Args:
            handler: 接收 HTTP 请求并返回模拟响应的函数。

        Returns:
            已安装的模拟 HTTP 客户端。
        """
        client = HTTP_CLIENT(transport=httpx.MockTransport(handler), follow_redirects=False)
        self.enterContext(patch('app.retrieval.web.httpx.Client', return_value=client))
        return client

    def test_local_private_and_non_http_addresses_are_rejected(self):
        """验证 URL 校验拒绝本地地址、私网地址及不支持的协议。"""
        for url in ('file:///etc/passwd', 'javascript:alert(1)', 'http://localhost/', 'http://a.local/',
                    'http://a.internal/', 'http://127.0.0.1/', 'http://10.0.0.1/', 'http://169.254.169.254/',
                    'http://[::1]/', 'https://example.org:8443/', 'https://' + 'user:pass@example.org/'):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    web.public_url(url, resolve=True)

    def test_dns_resolution_rejects_private_and_mixed_answers(self):
        """验证 DNS 校验拒绝含私有地址的解析结果。"""
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.168.0.2', 443))
        for answers in ([], [private], PUBLIC_DNS + [private]):
            with self.subTest(answers=answers):
                self.dns.return_value = answers
                with self.assertRaises(ValueError):
                    web.public_url('https://example.org/', resolve=True)

    def test_redirect_to_private_host_is_rejected_before_next_request(self):
        """验证读取公开网页时不会跟随指向私有地址的重定向。"""
        requests = []
        def handler(request):
            """记录请求并返回指向私有目标的重定向。

            Args:
                request: 当前 HTTP 请求，可用于访问应用状态。

            Returns:
                模拟的 HTTP 302 响应。
            """
            requests.append(str(request.url))
            return httpx.Response(302, headers={'location': 'http://127.0.0.1/admin'})
        self.client(handler)
        with self.assertRaises(ValueError):
            web.read_webpage('https://example.org/start')
        self.assertEqual(requests, ['https://example.org/start'])

    def test_public_redirect_is_resolved_and_extraction_strips_scripts(self):
        """验证公开重定向可正常读取，并从摘录中剔除脚本。"""
        def handler(request):
            """按请求路径模拟公开重定向和 HTML 页面。

            Args:
                request: 当前 HTTP 请求，可用于访问应用状态。

            Returns:
                模拟的重定向或 HTML 响应。
            """
            if request.url.path == '/start':
                return httpx.Response(302, headers={'location': '/article'})
            return httpx.Response(200, headers={'content-type': 'text/html; charset=utf-8'},
                content='<title>教学样例</title><nav>导航</nav><main>正文<script>不应输出</script><p>内容</p></main>'.encode())
        self.client(handler)
        result = web.read_webpage('https://example.org/start')
        self.assertEqual(result['kind'], 'web_page')
        self.assertEqual(result['url'], 'https://example.org/start')
        self.assertEqual(result['title'], '教学样例')
        self.assertEqual(result['text'], '正文\n内容')
        self.assertEqual(self.dns.call_count, 2)

    def test_unsupported_content_and_empty_body_fail(self):
        """验证不支持的内容类型和无正文响应会被拒绝。"""
        for content_type, content in (('application/pdf', b'%PDF'), ('text/html', b'<script>only script</script>')):
            with self.subTest(content_type=content_type):
                client = HTTP_CLIENT(transport=httpx.MockTransport(lambda request: httpx.Response(200,
                    headers={'content-type': content_type}, content=content)))
                try:
                    with patch('app.retrieval.web.httpx.Client', return_value=client):
                        with self.assertRaises(ValueError):
                            web.read_webpage('https://example.org/')
                finally:
                    client.close()

    def test_redirect_loop_is_bounded_and_text_is_truncated(self):
        """验证连续重定向和提取的正文都有上限。"""
        requests = []
        def handler(request):
            """记录请求并生成持续重定向的响应。

            Args:
                request: 当前 HTTP 请求，可用于访问应用状态。

            Returns:
                要求客户端继续跳转的模拟 HTTP 302 响应。
            """
            requests.append(request)
            return httpx.Response(302, headers={'location': '/again'})
        self.client(handler)
        with self.assertRaisesRegex(ValueError, 'redirects'):
            web.read_webpage('https://example.org/')
        self.assertEqual(len(requests), 5)
        client = HTTP_CLIENT(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            headers={'content-type': 'text/plain'}, content=b'A' * 8000)))
        try:
            with patch('app.retrieval.web.httpx.Client', return_value=client):
                self.assertEqual(len(web.read_webpage('https://example.org/')['text']), 4000)
        finally:
            client.close()

    def test_search_filters_private_urls_and_bounds_snippets(self):
        """验证搜索结果过滤私有地址并限制摘要长度。"""
        with patch('ddgs.DDGS') as ddgs:
            ddgs.return_value.text.return_value = [
                {'href': 'http://127.0.0.1/', 'title': 'private', 'body': 'hidden'},
                {'href': 'https://example.org/', 'title': '公开资料', 'body': 'A' * 2000},
                {'href': 'https://example.org/empty', 'body': ''},
            ]
            result = web.search_web('课程补充')
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['text']), 900)
        self.assertEqual(result[0]['kind'], 'web_snippet')
        self.assertEqual(ddgs.return_value.text.call_args.kwargs['max_results'], 3)
