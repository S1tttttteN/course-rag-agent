"""Model factory checks without network or access to local credentials."""
from pathlib import Path
import os
import runpy
import unittest
from unittest.mock import patch


MODELS = Path(__file__).resolve().parents[1] / 'app/core/models.py'


class ModelConfigTests(unittest.TestCase):
    def load(self, environment):
        with patch.dict(os.environ, environment, clear=True), patch('dotenv.load_dotenv'):
            return runpy.run_path(str(MODELS))

    def test_no_env_requires_explicit_provider_and_local_model(self):
        config = self.load({})
        self.assertEqual(config['CHAT_PROVIDER'], '')
        self.assertEqual(config['EMBEDDING_PROVIDER'], '')
        self.assertEqual(config['RERANK_PROVIDER'], 'none')
        self.assertFalse(config['ANSWER_REVIEW'])
        self.assertTrue(config['WEB_SEARCH_ENABLED'])
        self.assertEqual(config['RUN_TIMEOUT'], 180)
        with self.assertRaisesRegex(RuntimeError, 'CHAT_PROVIDER'):
            config['get_llm']()
        with self.assertRaisesRegex(RuntimeError, 'EMBEDDING_PROVIDER'):
            config['get_embeddings']()
        config = self.load({'CHAT_PROVIDER': 'ollama', 'CHAT_MODEL': 'fixture-chat-model', 'RUN_TIMEOUT': ''})
        with patch('langchain_ollama.ChatOllama') as ollama, patch('langchain_openai.ChatOpenAI') as cloud:
            config['get_llm']()
        cloud.assert_not_called()
        self.assertEqual(ollama.call_args.kwargs['model'], 'fixture-chat-model')
        self.assertFalse(ollama.call_args.kwargs['reasoning'])
        self.assertEqual(config['RUN_TIMEOUT'], 600)
        config = self.load({'CHAT_PROVIDER': 'ollama'})
        with self.assertRaisesRegex(RuntimeError, 'CHAT_MODEL'):
            config['get_llm']()
        config = self.load({'EMBEDDING_PROVIDER': 'ollama'})
        with self.assertRaisesRegex(RuntimeError, 'EMBEDDING_MODEL'):
            config['get_embeddings']()

    def test_cloud_is_explicit_and_missing_key_fails_without_fallback(self):
        config = self.load({'CHAT_PROVIDER': 'deepseek'})
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''}), patch('langchain_ollama.ChatOllama') as ollama:
            with self.assertRaisesRegex(RuntimeError, 'DEEPSEEK_API_KEY'):
                config['get_llm']()
        ollama.assert_not_called()
