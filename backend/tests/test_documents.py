"""Version publication, course isolation and worker recovery with real local indexes."""
import hashlib
import os
from threading import Event
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import select

from test_courses import app
from app.core.config import UPLOAD_DIR
from app.db.models import CleanupJob, DocumentVersion
from app.db.session import SessionLocal
from app.jobs.indexing import cleanup_job, process_version, recover
from app.retrieval import index
from app.retrieval.loader import load_and_split


class FakeEmbeddings:
    """Deterministic vectors for testing storage, never an alternative product model."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [0.1 + digest[position] / 255 for position in range(3)]


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.course = self.client.post('/api/courses', json={'name': '文档生命周期验证'}).json()['id']
        self.other = self.client.post('/api/courses', json={'name': '另一门课'}).json()['id']
        self.prefix = f'/api/courses/{self.course}/documents'

    def tearDown(self):
        for course in (self.course, self.other):
            self.client.delete(f'/api/courses/{course}')
        with SessionLocal() as session:
            jobs = list(session.scalars(select(CleanupJob.id)))
        for job in jobs:
            cleanup_job(job)
        self.client_context.__exit__(None, None, None)

    def upload(self, content='进程是资源分配的基本单位。线程是调度的基本单位。', filename='课程笔记.md'):
        response = self.client.post(self.prefix, files={'file': (filename, content.encode(), 'text/plain')})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def process(self, document):
        with patch('app.jobs.indexing.get_embeddings', return_value=FakeEmbeddings()), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
            process_version(document['latest']['id'], Event())
        return next(item for item in self.client.get(self.prefix).json() if item['id'] == document['id'])

    def test_index_stays_embedded_and_no_chroma_http_api_is_exposed(self):
        from chromadb.api.rust import RustBindingsAPI
        index.close()
        with patch.dict(os.environ, {'CHROMA_API_IMPL': 'chromadb.api.fastapi.FastAPI'}):
            self.assertIsInstance(index.client()._server, RustBindingsAPI)
        self.assertFalse(any(getattr(route, 'path', '').startswith('/api/v2') for route in app.routes))

    def test_publication_preview_and_course_scoped_search(self):
        document = self.upload()
        self.assertFalse(document['usable'])
        self.assertEqual(self.client.get(f"{self.prefix}/{document['id']}/preview").status_code, 409)
        document = self.process(document)
        self.assertTrue(document['usable'])
        preview = self.client.get(f"{self.prefix}/{document['id']}/preview").json()
        self.assertIn('进程', preview['chunks'][0]['text'])
        self.assertEqual(preview['source'], '课程笔记.md')
        with patch('app.retrieval.search.get_embeddings', return_value=FakeEmbeddings()), patch('app.retrieval.search.embedding_profile', return_value='test:3'):
            results = self.client.post(f'/api/courses/{self.course}/search', json={'query': '进程'}).json()
        self.assertTrue(results['trace']['semantic'])
        self.assertTrue(results['trace']['keyword'])
        self.assertEqual(results['results'][0]['document_id'], document['id'])
        self.assertEqual(self.client.get(f"/api/courses/{self.other}/documents/{document['id']}/preview").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/courses/{self.other}/documents/{document['id']}").status_code, 404)
        self.assertEqual(self.client.post(f'/api/courses/{self.other}/search', json={'query': '进程', 'semantic': False}).json()['results'], [])

    def test_failed_replacement_keeps_old_version_then_retry_publishes(self):
        original = self.process(self.upload())
        replacement = self.client.put(f"{self.prefix}/{original['id']}/file", files={'file': ('新版.md', '虚拟内存采用分页机制。'.encode())}).json()
        with patch('app.jobs.indexing.get_embeddings', side_effect=RuntimeError('offline fixture')), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
            process_version(replacement['latest']['id'], Event())
        failed = self.client.get(self.prefix).json()[0]
        self.assertEqual(failed['latest']['status'], 'failed')
        self.assertEqual(failed['active_version_id'], original['active_version_id'])
        self.assertTrue(failed['usable'])
        found = self.client.post(f'/api/courses/{self.course}/search', json={'query': '进程', 'semantic': False}).json()['results']
        self.assertTrue(found)
        retried = self.client.post(f"{self.prefix}/{original['id']}/retry")
        self.assertEqual(retried.status_code, 202)
        published = self.process(retried.json())
        self.assertEqual(published['active_version_number'], 2)
        self.assertEqual(published['id'], original['id'])
        self.assertEqual(self.client.post(f'/api/courses/{self.course}/search', json={'query': '进程', 'semantic': False}).json()['results'], [])
        self.assertEqual(index.collection(self.course, 'test:3').count(), published['chunk_count'])

    def test_delete_immediately_hides_data_and_cleanup_removes_files(self):
        document = self.process(self.upload())
        self.assertEqual(self.client.delete(f"{self.prefix}/{document['id']}").status_code, 204)
        self.assertEqual(self.client.post(f'/api/courses/{self.course}/search', json={'query': '进程', 'semantic': False}).json()['results'], [])
        with SessionLocal() as session:
            job = session.scalar(select(CleanupJob.id).where(CleanupJob.document_id == document['id']))
        cleanup_job(job)
        self.assertFalse((UPLOAD_DIR / self.course / document['id']).exists())
        self.assertEqual(index.collection(self.course, 'test:3').count(), 0)

    def test_empty_unsupported_duplicate_and_pending_replace(self):
        self.assertEqual(self.client.post(self.prefix, files={'file': ('空.txt', b'')}).status_code, 400)
        self.assertEqual(self.client.post(self.prefix, files={'file': ('图.png', b'not text')}).status_code, 400)
        document = self.upload()
        self.assertEqual(self.client.post(self.prefix, files={'file': ('课程笔记.md', b'duplicate')}).status_code, 409)
        self.assertEqual(self.client.put(f"{self.prefix}/{document['id']}/file", files={'file': ('v2.md', b'text')}).status_code, 409)
        self.assertEqual(self.client.post(f"{self.prefix}/{document['id']}/retry").status_code, 409)

    def test_whitespace_is_failed_without_becoming_usable(self):
        document = self.process(self.upload('  \n\n  '))
        self.assertEqual(document['latest']['status'], 'failed')
        self.assertIn('正文', document['latest']['error'])
        self.assertFalse(document['usable'])

    def test_recovery_requeues_processing_and_cleans_interrupted_vectors(self):
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
        document = self.upload()
        client = self.client
        path = f"{self.prefix}/{document['id']}"
        class DeleteDuringEmbedding(FakeEmbeddings):
            def embed_documents(self, texts):
                client.delete(path)
                return super().embed_documents(texts)
        with patch('app.jobs.indexing.get_embeddings', return_value=DeleteDuringEmbedding()), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
            process_version(document['latest']['id'], Event())
        self.assertEqual(self.client.get(self.prefix).json(), [])
        self.assertEqual(index.collection(self.course, 'test:3').count(), 0)

    def test_oversize_and_filename_sanitization(self):
        document = self.upload('正文', '../../合法笔记.txt')
        self.assertEqual(document['name'], '合法笔记.txt')
        response = self.client.post(self.prefix, files={'file': ('too-big.txt', b'a' * (20 * 1024 * 1024 + 1))})
        self.assertEqual(response.status_code, 413)

    def test_course_delete_cascades_documents_and_cleanup_is_scoped(self):
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
    def test_html_gbk_word_and_multi_page_pdf(self):
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
    def test_upload_is_processed_by_background_worker(self):
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
