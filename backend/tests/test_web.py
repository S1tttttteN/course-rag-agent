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
    def setUp(self):
        self.dns = self.enterContext(patch('app.retrieval.web.socket.getaddrinfo', return_value=PUBLIC_DNS))

    def client(self, handler):
        client = HTTP_CLIENT(transport=httpx.MockTransport(handler), follow_redirects=False)
        self.enterContext(patch('app.retrieval.web.httpx.Client', return_value=client))
        return client

    def test_local_private_and_non_http_addresses_are_rejected(self):
        for url in ('file:///etc/passwd', 'javascript:alert(1)', 'http://localhost/', 'http://a.local/',
                    'http://a.internal/', 'http://127.0.0.1/', 'http://10.0.0.1/', 'http://169.254.169.254/',
                    'http://[::1]/', 'https://example.org:8443/', 'https://' + 'user:pass@example.org/'):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    web.public_url(url, resolve=True)

    def test_dns_resolution_rejects_private_and_mixed_answers(self):
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.168.0.2', 443))
        for answers in ([], [private], PUBLIC_DNS + [private]):
            with self.subTest(answers=answers):
                self.dns.return_value = answers
                with self.assertRaises(ValueError):
                    web.public_url('https://example.org/', resolve=True)

    def test_redirect_to_private_host_is_rejected_before_next_request(self):
        requests = []
        def handler(request):
            requests.append(str(request.url))
            return httpx.Response(302, headers={'location': 'http://127.0.0.1/admin'})
        self.client(handler)
        with self.assertRaises(ValueError):
            web.read_webpage('https://example.org/start')
        self.assertEqual(requests, ['https://example.org/start'])

    def test_public_redirect_is_resolved_and_extraction_strips_scripts(self):
        def handler(request):
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
        requests = []
        def handler(request):
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
