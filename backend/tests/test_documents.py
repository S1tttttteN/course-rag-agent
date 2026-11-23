"""Version publication, course isolation and worker recovery with real local indexes."""
import hashlib
from io import BytesIO
import os
from threading import Event
import time
import unittest
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4
from zipfile import ZipFile

import httpx
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from openai import APIStatusError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import select

from test_courses import app
from app.retrieval.search import search, search_course_material
from app.core.config import UPLOAD_DIR
from app.db.models import CleanupJob, DocumentVersion
from app.db.session import SessionLocal
from app.jobs.indexing import cleanup_job, process_version, recover
from app.retrieval import index
from app.retrieval.loader import load_and_split
from app.services.documents import raw_path


class FakeEmbeddings:
    """为存储测试提供确定性的三维嵌入向量。"""

    def embed_documents(self, texts):
        """按输入顺序生成测试资料向量。

        Args:
            texts: 要编码的资料文本列表。

        Returns:
            与输入一一对应的三维向量列表。
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        """使用文本摘要生成可复现的测试向量。

        Args:
            text: 待处理的文本。

        Returns:
            由摘要前三个字节映射出的三维向量。
        """
        digest = hashlib.sha256(text.encode()).digest()
        return [0.1 + digest[position] / 255 for position in range(3)]


class DocumentTests(unittest.TestCase):
    """验证课程资料上传、索引、版本发布和清理。"""

    def setUp(self):
        """启动隔离的测试客户端并创建两门测试课程。"""
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.course = self.client.post('/api/courses', json={'name': '文档生命周期验证'}).json()['id']
        self.other = self.client.post('/api/courses', json={'name': '另一门课'}).json()['id']
        self.prefix = f'/api/courses/{self.course}/documents'

    def tearDown(self):
        """删除测试课程并完成待处理的存储清理。"""
        for course in (self.course, self.other):
            self.client.delete(f'/api/courses/{course}')
        with SessionLocal() as session:
            jobs = list(session.scalars(select(CleanupJob.id)))
        for job in jobs:
            cleanup_job(job)
        self.client_context.__exit__(None, None, None)

    def upload(self, content='进程是资源分配的基本单位。线程是调度的基本单位。', filename='课程笔记.md'):
        """上传一份测试资料并验证请求已接受。

        Args:
            content: 要编码为 UTF-8 的资料正文。
            filename: 上传文件的原始名称。

        Returns:
            接口返回的新资料和版本信息。
        """
        response = self.client.post(self.prefix, files={'file': (filename, content.encode(), 'text/plain')})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def process(self, document):
        """使用固定向量同步处理资料的最新版本。

        Args:
            document: 包含待处理版本 ID 的资料响应。

        Returns:
            刷新后的资料摘要。
        """
        with patch('app.jobs.indexing.get_embeddings', return_value=FakeEmbeddings()), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
            process_version(document['latest']['id'], Event())
        return next(item for item in self.client.get(self.prefix).json() if item['id'] == document['id'])

    def test_index_stays_embedded_and_no_chroma_http_api_is_exposed(self):
        """验证向量索引始终使用嵌入式客户端，且未挂载 Chroma HTTP 路由。"""
        from chromadb.api.rust import RustBindingsAPI
        index.close()
        with patch.dict(os.environ, {'CHROMA_API_IMPL': 'chromadb.api.fastapi.FastAPI'}):
            self.assertIsInstance(index.client()._server, RustBindingsAPI)
        self.assertFalse(any(getattr(route, 'path', '').startswith('/api/v2') for route in app.routes))

    def test_publication_preview_and_course_scoped_search(self):
        """验证发布后的资料可预览，并且搜索限定在当前课程。"""
        document = self.upload()
        self.assertFalse(document['usable'])
        self.assertEqual(self.client.get(f"{self.prefix}/{document['id']}/preview").status_code, 409)
        document = self.process(document)
        self.assertTrue(document['usable'])
        preview = self.client.get(f"{self.prefix}/{document['id']}/preview").json()
        self.assertIn('进程', preview['chunks'][0]['text'])
        self.assertEqual(preview['source'], '课程笔记.md')
        with patch('app.retrieval.search.get_embeddings', return_value=FakeEmbeddings()), patch('app.retrieval.search.embedding_profile', return_value='test:3'):
            results = search_course_material(self.course, '进程')
        self.assertTrue(results['trace']['semantic'])
        self.assertEqual(self.client.post(f'/api/courses/{self.course}/search', json={'query': '进程'}).status_code, 404)
        self.assertTrue(results['trace']['keyword'])
        self.assertEqual(results['results'][0]['document_id'], document['id'])
        self.assertEqual(self.client.get(f"/api/courses/{self.other}/documents/{document['id']}/preview").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/courses/{self.other}/documents/{document['id']}").status_code, 404)
        self.assertEqual(search(self.other, '进程', semantic=False)['results'], [])

    def test_product_retrieval_falls_back_and_uses_configured_reranker(self):
        """向量故障时继续用关键词，重排服务可用时自动执行。"""
        document = self.process(self.upload())
        with (
            patch('app.retrieval.search.embedding_profile', return_value='test:3'),
            patch('app.retrieval.search.get_embeddings', side_effect=RuntimeError('offline')),
            patch('app.retrieval.search.is_available', return_value=True),
            patch('app.retrieval.search.rerank_documents', side_effect=lambda query, docs, limit: (docs, '')) as rerank,
        ):
            result = search_course_material(self.course, '进程')
            self.assertTrue(result['trace']['keyword'])
            self.assertFalse(result['trace']['semantic'])
            self.assertTrue(result['trace']['reranked'])
            self.assertEqual(result['results'][0]['document_id'], document['id'])
            rerank.return_value = ([], 'unavailable')
            rerank.side_effect = None
            fallback = search_course_material(self.course, '进程')
        self.assertFalse(fallback['trace']['reranked'])
        self.assertIn('重排暂不可用', fallback['trace']['note'])
        self.assertEqual(fallback['results'], result['results'])

    def test_failed_replacement_keeps_old_version_then_retry_publishes(self):
        """验证替换失败时保留原版本，重试成功后才发布新版本。"""
        original = self.process(self.upload())
        replacement = self.client.put(f"{self.prefix}/{original['id']}/file", files={'file': ('新版.md', '虚拟内存采用分页机制。'.encode())}).json()
        with patch('app.jobs.indexing.get_embeddings', side_effect=RuntimeError('offline fixture')), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
            process_version(replacement['latest']['id'], Event())
        failed = self.client.get(self.prefix).json()[0]
        self.assertEqual(failed['latest']['status'], 'failed')
        self.assertEqual(failed['active_version_id'], original['active_version_id'])
        self.assertTrue(failed['usable'])
        found = search(self.course, '进程', semantic=False)['results']
        self.assertTrue(found)
        retried = self.client.post(f"{self.prefix}/{original['id']}/retry")
        self.assertEqual(retried.status_code, 202)
        published = self.process(retried.json())
        self.assertEqual(published['active_version_number'], 2)
        self.assertEqual(published['id'], original['id'])
        self.assertEqual(search(self.course, '进程', semantic=False)['results'], [])
        self.assertEqual(index.collection(self.course, 'test:3').count(), published['chunk_count'])

    def test_pdf_original_enforces_ownership_and_survives_cross_format_replacement(self):
        """验证 PDF 原件可按版本读取，且不跨课程、资料或删除边界。"""
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({
            NameObject('/Subtype'): NameObject('/Type1'),
            NameObject('/BaseFont'): NameObject('/Helvetica'),
        })
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 10 100 Td (Original page) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        output = BytesIO()
        writer.write(output)
        data = output.getvalue()

        filename = '课程讲义.PDF'
        upload = self.client.post(self.prefix, files={'file': (filename, data)})
        self.assertEqual(upload.status_code, 202)
        document = upload.json()
        document_id, version_id = document['id'], document['latest']['id']
        url = f'{self.prefix}/{document_id}/versions/{version_id}/pdf'
        self.assertIsNone(document['active_version_filename'])
        original = self.client.get(url)
        self.assertEqual(original.status_code, 200)
        self.assertEqual(original.content, data)
        self.assertEqual(original.headers['content-type'], 'application/pdf')
        self.assertIn("inline; filename*=utf-8''" + quote(filename), original.headers['content-disposition'])
        partial = self.client.get(url, headers={'Range': 'bytes=0-7'})
        self.assertEqual(partial.status_code, 206)
        self.assertEqual(partial.headers['content-range'], f'bytes 0-7/{len(data)}')
        self.assertEqual(partial.content, data[:8])

        self.assertEqual(self.process(document)['active_version_filename'], filename)
        replacement = self.client.put(f'{self.prefix}/{document_id}/file', files={'file': ('新版.md', b'New text')})
        self.assertEqual(replacement.status_code, 202)
        self.assertEqual(replacement.json()['active_version_filename'], filename)
        self.assertEqual(self.process(replacement.json())['active_version_filename'], '新版.md')
        self.assertEqual(self.client.get(url).content, data)
        replacement_id = replacement.json()['latest']['id']
        self.assertEqual(self.client.get(f'{self.prefix}/{document_id}/versions/{replacement_id}/pdf').status_code, 415)
        self.assertEqual(self.client.get(url.replace(self.course, self.other, 1)).status_code, 404)
        self.assertEqual(self.client.get(url.replace(document_id, self.upload()['id'], 1)).status_code, 404)
        self.assertEqual(self.client.get(url.replace(version_id, str(uuid4()))).status_code, 404)

        path = raw_path(self.course, document_id, version_id, filename)
        path.unlink()
        self.assertEqual(self.client.get(url).status_code, 404)
        path.write_bytes(data)
        self.assertEqual(self.client.delete(f'{self.prefix}/{document_id}').status_code, 204)
        self.assertTrue(path.exists())  # Access ends before background cleanup removes the bytes.
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_text_original_reads_queued_and_historical_versions(self):
        """验证按版本读取全文，并在替换、删除和跨课程访问时校验原件。"""
        filename = '调度.TXT'
        draft = self.upload('原始讲义\n  轮转调度', filename)
        document_id, version_id = draft['id'], draft['latest']['id']
        base = f'{self.prefix}/{document_id}/versions'
        old_url = f'{base}/{version_id}/text'
        expected = {'text': '原始讲义\n  轮转调度'}
        self.assertEqual(self.client.get(old_url).json(), expected)
        self.assertEqual(self.client.get(old_url).headers['cache-control'], 'no-store')
        self.process(draft)

        replacement = self.client.put(
            f'{self.prefix}/{document_id}/file',
            files={'file': ('新版.html', b'<h1>New version</h1>', 'text/html')},
        )
        self.assertEqual(replacement.status_code, 202)
        published = self.process(replacement.json())
        self.assertEqual(published['active_version_filename'], '新版.html')
        self.assertEqual(self.client.get(old_url).json(), expected)
        self.assertEqual(self.client.get(f"{base}/{published['active_version_id']}/text").status_code, 415)
        self.assertEqual(self.client.get(old_url.replace(self.course, self.other, 1)).status_code, 404)
        self.assertEqual(self.client.get(old_url.replace(document_id, self.upload()['id'], 1)).status_code, 404)
        self.assertEqual(self.client.get(f'{base}/{uuid4()}/text').status_code, 404)

        path = raw_path(self.course, document_id, version_id, filename)
        path.unlink()
        self.assertEqual(self.client.get(old_url).status_code, 404)
        path.write_text(expected['text'], encoding='utf-8')
        self.assertEqual(self.client.delete(f'{self.prefix}/{document_id}').status_code, 204)
        self.assertTrue(path.exists())
        self.assertEqual(self.client.get(old_url).status_code, 404)

    def test_text_original_uses_index_decoding_and_body_limit(self):
        """验证全文读取与索引的解码、换行及长度限制一致。"""
        text = '# 调度\r\n\t保留缩进\r课件'
        for encoding in ('utf-8', 'gb18030', 'utf-16', 'utf-32'):
            with self.subTest(encoding=encoding):
                response = self.client.post(
                    self.prefix, files={'file': (f'{encoding}.md', text.encode(encoding))}
                )
                self.assertEqual(response.status_code, 202)
                document = response.json()
                url = f"{self.prefix}/{document['id']}/versions/{document['latest']['id']}/text"
                result = self.client.get(url)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json(), {'text': '# 调度\n\t保留缩进\n课件'})

        for filename, data in (('invalid.txt', b'\xff\xfe\x00'), ('large.txt', b'a' * 2_000_001)):
            with self.subTest(filename=filename):
                response = self.client.post(self.prefix, files={'file': (filename, data)})
                self.assertEqual(response.status_code, 202)
                document = response.json()
                url = f"{self.prefix}/{document['id']}/versions/{document['latest']['id']}/text"
                self.assertEqual(self.client.get(url).status_code, 422)

    def test_rich_preview_reads_safe_historical_html(self):
        """验证 HTML 预览保留课件样式和链接，同时清理主动内容及外部资源。"""
        png = (
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8'
            '/x8AAwMCAO+jYxoAAAAASUVORK5CYII='
        )
        markup = f'''<!doctype html><html><head>
        <meta charset="gbk"><base href="https://blocked.invalid/">
        <meta http-equiv="refresh" content="0;url=https://blocked.invalid/">
        <style>body.lecture {{ color: purple; }}</style>
        <link rel="stylesheet" href="https://blocked.invalid/lecture.css"></head>
        <body class="lecture"><h1 id="调度">课程讲义</h1>
        <a href="#调度">本页目录</a><a href="https://example.test/page" ping="/api/courses">安全外链</a>
        <a href="../next.html">相对文件</a><a href="javascript:alert(1)">危险链接</a>
        <img src="data:image/png;base64,{png}" alt="内嵌图" onerror="alert(1)">
        <img src="https://blocked.invalid/pixel" srcset="https://blocked.invalid/pixel2" alt="远程图">
        <img src="data:image/png;base64,PHN2Zz4=" alt="伪装图片">
        <form action="https://blocked.invalid/"><input name="bad"></form>
        <iframe src="https://blocked.invalid/"></iframe><script>bad()</script>
        </body></html>'''
        response = self.client.post(self.prefix, files={'file': ('课件.HTM', markup.encode('gbk'), 'text/html')})
        self.assertEqual(response.status_code, 202)
        document = response.json()
        document_id, old_id = document['id'], document['latest']['id']
        base = f'{self.prefix}/{document_id}/versions'
        url = f'{base}/{old_id}/rich-preview'
        result = self.client.get(url)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers['cache-control'], 'no-store')
        preview = result.json()
        self.assertIn('body.lecture { color: purple; }', preview['css'])
        body = BeautifulSoup(preview['html'], 'html.parser')
        self.assertEqual(body.div.get('class'), ['lecture'])
        self.assertEqual(body.find('h1').get_text(), '课程讲义')
        self.assertEqual([a.get('href') for a in body.find_all('a')], ['#调度', 'https://example.test/page', None, None])
        self.assertEqual(body.find('img', alt='内嵌图').get('data-preview-image'), f'data:image/png;base64,{png}')
        self.assertIsNone(body.find('img', alt='伪装图片').get('data-preview-image'))
        self.assertNotIn('blocked.invalid', preview['html'])
        self.assertIsNone(body.select_one('script, form, iframe, meta, link, base, [onerror], [srcset], [ping]'))

        self.process(document)
        replacement = self.client.put(
            f'{self.prefix}/{document_id}/file', files={'file': ('新版.txt', b'new version')}
        )
        new_version = self.process(replacement.json())['active_version_id']
        self.assertEqual(self.client.get(f'{base}/{new_version}/rich-preview').status_code, 415)
        self.assertEqual(self.client.get(url).json(), preview)
        self.assertEqual(self.client.get(url.replace(self.course, self.other, 1)).status_code, 404)
        self.assertEqual(self.client.get(url.replace(document_id, self.upload()['id'], 1)).status_code, 404)
        self.assertEqual(self.client.delete(f'{self.prefix}/{document_id}').status_code, 204)
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_docx_preview_and_historical_original(self):
        """验证 DOCX 的结构预览、历史版本原件及损坏文件的错误处理。"""
        ns_word = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        ns_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
        types = "http://schemas.openxmlformats.org/package/2006/content-types"
        relationship = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        word = BytesIO()
        with ZipFile(word, "w") as package:
            package.writestr("[Content_Types].xml", f'''<Types xmlns="{types}">
              <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
              <Default Extension="xml" ContentType="application/xml"/>
              <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
              <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
            </Types>''')
            package.writestr("_rels/.rels", f'''<Relationships xmlns="{ns_rel}">
              <Relationship Id="r1" Type="{relationship}/officeDocument" Target="word/document.xml"/>
            </Relationships>''')
            package.writestr("word/_rels/document.xml.rels", f'''<Relationships xmlns="{ns_rel}">
              <Relationship Id="r2" Type="{relationship}/styles" Target="styles.xml"/>
            </Relationships>''')
            package.writestr("word/styles.xml", f'''<w:styles xmlns:w="{ns_word}">
              <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
            </w:styles>''')
            package.writestr("word/document.xml", f'''<w:document xmlns:w="{ns_word}"><w:body>
              <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>作业要求</w:t></w:r></w:p>
              <w:tbl><w:tr><w:tc><w:p><w:r><w:t>表格正文</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
            </w:body></w:document>''')
        response = self.client.post(self.prefix, files={'file': ('作业.docx', word.getvalue())})
        self.assertEqual(response.status_code, 202)
        document = response.json()
        url = f"{self.prefix}/{document['id']}/versions/{document['latest']['id']}/rich-preview"
        preview = self.client.get(url)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.json()['css'], '')
        body = BeautifulSoup(preview.json()['html'], 'html.parser')
        self.assertEqual(body.h1.get_text(), '作业要求')
        self.assertEqual(body.select_one('table tr td').get_text(), '表格正文')

        original = url.removesuffix('/rich-preview') + '/docx'
        result = self.client.get(original)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content, word.getvalue())
        self.assertEqual(result.headers['cache-control'], 'no-store')
        self.assertEqual(result.headers['x-content-type-options'], 'nosniff')
        self.assertEqual(self.client.get(original.replace(self.course, self.other, 1)).status_code, 404)

        self.process(document)
        replaced = self.client.put(
            f"{self.prefix}/{document['id']}/file", files={'file': ('新版.txt', b'new version')}
        )
        new_id = self.process(replaced.json())['active_version_id']
        self.assertEqual(self.client.get(original.replace(document['latest']['id'], new_id)).status_code, 415)
        self.assertEqual(self.client.get(original).content, word.getvalue())
        self.assertEqual(self.client.delete(f"{self.prefix}/{document['id']}").status_code, 204)
        self.assertEqual(self.client.get(original).status_code, 404)

        broken = self.upload('invalid DOCX', 'invalid.docx')
        invalid = f"{self.prefix}/{broken['id']}/versions/{broken['latest']['id']}/rich-preview"
        self.assertEqual(self.client.get(invalid).status_code, 422)
        self.assertEqual(self.client.get(invalid.removesuffix('/rich-preview') + '/docx').status_code, 422)

    def test_delete_immediately_hides_data_and_cleanup_removes_files(self):
        """验证删除立即隐藏资料，后台清理随后移除存储文件。"""
        document = self.process(self.upload())
        self.assertEqual(self.client.delete(f"{self.prefix}/{document['id']}").status_code, 204)
        self.assertEqual(search(self.course, '进程', semantic=False)['results'], [])
        with SessionLocal() as session:
            job = session.scalar(select(CleanupJob.id).where(CleanupJob.document_id == document['id']))
        cleanup_job(job)
        self.assertFalse((UPLOAD_DIR / self.course / document['id']).exists())
        self.assertEqual(index.collection(self.course, 'test:3').count(), 0)

    def test_empty_unsupported_duplicate_and_pending_replace(self):
        """验证上传拒绝空文件、不支持的格式和同名资料，处理中版本也不能替换。"""
        self.assertEqual(self.client.post(self.prefix, files={'file': ('空.txt', b'')}).status_code, 400)
        self.assertEqual(self.client.post(self.prefix, files={'file': ('图.png', b'not text')}).status_code, 400)
        document = self.upload()
        self.assertEqual(self.client.post(self.prefix, files={'file': ('课程笔记.md', b'duplicate')}).status_code, 409)
        self.assertEqual(self.client.put(f"{self.prefix}/{document['id']}/file", files={'file': ('v2.md', b'text')}).status_code, 409)
        self.assertEqual(self.client.post(f"{self.prefix}/{document['id']}/retry").status_code, 409)

    def test_whitespace_is_failed_without_becoming_usable(self):
        """验证仅含空白的文件索引失败且不会发布。"""
        document = self.process(self.upload('  \n\n  '))
        self.assertEqual(document['latest']['status'], 'failed')
        self.assertIn('正文', document['latest']['error'])
        self.assertFalse(document['usable'])

    def test_embedding_payment_error_is_actionable_and_can_retry(self):
        """验证余额不足时显示可操作的提示，服务恢复后可用同一原件重试。"""
        document = self.upload()
        response = httpx.Response(402, request=httpx.Request('POST', 'https://example.test/embeddings'))
        error = APIStatusError('private provider response', response=response, body={'code': 30001})
        with patch.object(FakeEmbeddings, 'embed_documents', side_effect=error):
            failed = self.process(document)
        self.assertEqual(failed['latest']['status'], 'failed')
        self.assertIn('余额不足', failed['latest']['error'])
        self.assertIn('402', failed['latest']['error'])
        self.assertNotIn('private provider response', failed['latest']['error'])
        self.assertFalse(failed['usable'])

        retried = self.client.post(f"{self.prefix}/{document['id']}/retry")
        self.assertEqual(retried.status_code, 202)
        published = self.process(retried.json())
        self.assertEqual(published['active_version_id'], document['latest']['id'])
        self.assertTrue(published['usable'])

    def test_recovery_requeues_processing_and_cleans_interrupted_vectors(self):
        """验证恢复流程重新排队处理中版本并清除未发布向量。"""
        document = self.upload()
        version_id = document['latest']['id']
        with SessionLocal() as session:
            version = session.get(DocumentVersion, version_id)
            version.status = 'processing'
            session.commit()
        store = index.collection(self.course, 'test:3')
        store.upsert(ids=[f'{version_id}:0'], embeddings=[[1., 1., 1.]], metadatas=[{'version_id': version_id, 'document_id': document['id']}])
        recover()
        self.assertEqual(self.client.get(self.prefix).json()[0]['latest']['status'], 'queued')
        self.assertEqual(store.count(), 0)
        self.assertTrue(self.process(document)['usable'])

    def test_delete_during_embedding_cannot_publish_orphan(self):
        """验证资料在嵌入期间被删除时，索引任务不会发布孤立片段。"""
        document = self.upload()
        client = self.client
        path = f"{self.prefix}/{document['id']}"
        class DeleteDuringEmbedding(FakeEmbeddings):
            """模拟在生成资料嵌入时删除该资料。"""

            def embed_documents(self, texts):
                """触发删除后返回确定性的测试向量。

                Args:
                    texts: 传给索引器的资料文本列表。

                Returns:
                    与输入一一对应的测试向量列表。
                """
                client.delete(path)
                return super().embed_documents(texts)
        with patch('app.jobs.indexing.get_embeddings', return_value=DeleteDuringEmbedding()), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
            process_version(document['latest']['id'], Event())
        self.assertEqual(self.client.get(self.prefix).json(), [])
        self.assertEqual(index.collection(self.course, 'test:3').count(), 0)

    def test_oversize_and_filename_sanitization(self):
        """验证上传文件大小上限和文件名处理。"""
        document = self.upload('正文', '../../合法笔记.txt')
        self.assertEqual(document['name'], '合法笔记.txt')
        response = self.client.post(self.prefix, files={'file': ('too-big.txt', b'a' * (20 * 1024 * 1024 + 1))})
        self.assertEqual(response.status_code, 413)

    def test_course_delete_cascades_documents_and_cleanup_is_scoped(self):
        """验证删除课程级联移除其资料，清理任务仅处理该课程。"""
        document = self.process(self.upload())
        other_prefix = f'/api/courses/{self.other}/documents'
        other_document = self.client.post(other_prefix, files={'file': ('同名.md', '网络协议'.encode())}).json()
        self.assertEqual(self.client.delete(f'/api/courses/{self.course}').status_code, 204)
        self.assertEqual(self.client.get(self.prefix).status_code, 404)
        self.assertEqual(self.client.get(other_prefix).json()[0]['id'], other_document['id'])
        with SessionLocal() as session:
            self.assertIsNone(session.get(DocumentVersion, document['latest']['id']))
            job = session.scalar(select(CleanupJob.id).where(CleanupJob.course_id == self.course))
        cleanup_job(job)
        self.assertFalse((UPLOAD_DIR / self.course).exists())
        self.assertTrue((UPLOAD_DIR / self.other).exists())
        self.assertFalse(any(item.name.startswith(f"course_{self.course.replace('-', '')}_") for item in index.client().list_collections()))

    def test_failed_cleanup_remains_durable_and_can_retry(self):
        """验证失败的存储清理任务保留在数据库中并可重试。"""
        document = self.process(self.upload())
        self.client.delete(f"{self.prefix}/{document['id']}")
        with SessionLocal() as session:
            job_id = session.scalar(select(CleanupJob.id).where(CleanupJob.document_id == document['id']))
        with patch('app.jobs.indexing.index.remove_document', side_effect=RuntimeError('fixture')):
            cleanup_job(job_id)
        with SessionLocal() as session:
            self.assertEqual(session.get(CleanupJob, job_id).attempts, 1)
        cleanup_job(job_id)
        with SessionLocal() as session:
            self.assertIsNone(session.get(CleanupJob, job_id))


class ParserTests(unittest.TestCase):
    """验证支持的资料解析路径及来源元数据。"""

    def test_text_with_unicode_bom(self):
        """验证 UTF-16/32 字节序标记的识别，并保留正文及换行。"""
        from codecs import BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE
        from pathlib import Path
        from tempfile import TemporaryDirectory

        content = '操作系统\r\n线程调度'
        with TemporaryDirectory() as directory:
            for suffix, bom, encoding in (
                ('.txt', BOM_UTF16_LE, 'utf-16-le'),
                ('.md', BOM_UTF16_BE, 'utf-16-be'),
                ('.txt', BOM_UTF32_LE, 'utf-32-le'),
                ('.md', BOM_UTF32_BE, 'utf-32-be'),
            ):
                with self.subTest(encoding=encoding):
                    path = Path(directory) / f'note{suffix}'
                    path.write_bytes(bom + content.encode(encoding))
                    chunks = load_and_split(path, path.name)
                    self.assertEqual(chunks[0].page_content, '操作系统\n线程调度')

    def test_html_gbk_word_and_multi_page_pdf(self):
        """验证 GBK HTML、Word 和多页 PDF 的正文与页码解析。"""
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as directory:
            path = Path(directory)
            html = path / 'note.html'
            html.write_bytes('<meta charset="gbk"><style>hidden</style><p>操作系统正文</p><script>secret</script>'.encode('gbk'))
            content = load_and_split(html, html.name)[0].page_content
            self.assertIn('操作系统正文', content)
            self.assertNotIn('secret', content)
            word = path / 'note.docx'
            with ZipFile(word, 'w') as archive:
                archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Word text content</w:t></w:r></w:p></w:body></w:document>')
            self.assertIn('Word text', load_and_split(word, word.name)[0].page_content)
            writer = PdfWriter()
            for text in ('Page one', 'Page two'):
                page = writer.add_blank_page(width=300, height=300)
                font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
                page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
                stream = DecodedStreamObject()
                stream.set_data(f'BT /F1 12 Tf 10 100 Td ({text}) Tj ET'.encode())
                page[NameObject('/Contents')] = writer._add_object(stream)
            pdf = path / 'note.pdf'
            with pdf.open('wb') as output:
                writer.write(output)
            chunks = load_and_split(pdf, pdf.name)
            self.assertEqual({chunk.metadata['page'] for chunk in chunks}, {1, 2})
            self.assertIn('Page two', chunks[-1].page_content)


class WorkerTests(unittest.TestCase):
    """验证资料索引后台线程的自动处理。"""

    def test_upload_is_processed_by_background_worker(self):
        """验证上传资料会被运行中的索引线程自动发布。"""
        app.state.indexing_enabled = True
        try:
            with patch('app.jobs.indexing.get_embeddings', return_value=FakeEmbeddings()), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
                with TestClient(app) as client:
                    course_id = client.post('/api/courses', json={'name': '自动建库验证'}).json()['id']
                    prefix = f'/api/courses/{course_id}/documents'
                    uploaded = client.post(prefix, files={'file': ('自动.md', '操作系统自动建库。'.encode())})
                    self.assertEqual(uploaded.status_code, 202)
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        document = client.get(prefix).json()[0]
                        if document['latest']['status'] in {'ready', 'failed'}:
                            break
                        time.sleep(0.05)
                    self.assertEqual(document['latest']['status'], 'ready', document)
                    self.assertTrue(document['usable'])
                    client.delete(f'/api/courses/{course_id}')
        finally:
            app.state.indexing_enabled = False
