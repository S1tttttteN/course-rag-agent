"""Opt-in real local-model smoke check. No app data, cloud calls, or web access."""
import asyncio
import argparse
import json
import os

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


async def main(model_name):
    """使用本地 Ollama 模型验证一次工具调用和最终回答。

    Args:
        model_name: 本机已安装的 Ollama 模型标签。

    Raises:
        RuntimeError: 模型未完成预期工具调用或未给出预期结果。
        TimeoutError: 冒烟检查超过 240 秒。
    """
    calls = []

    @tool
    def lookup_demo_course(topic: str) -> str:
        """查询虚构课程实验的观察窗口；真实值只存在于此工具中。

        Args:
            topic: 需要查找的虚构实验主题。

        Returns:
            包含预设观察窗口的中文句子。
        """
        calls.append(topic)
        return '虚构实验的观察窗口为 17 分钟。'

    model = ChatOllama(model=model_name, base_url='http://127.0.0.1:11434', reasoning=False,
                       temperature=0, num_ctx=8192, num_predict=700, client_kwargs={'timeout': 180})
    agent = create_agent(model=model, tools=[lookup_demo_course],
        system_prompt='你是测试助手。涉及虚构实验数据必须先调用 lookup_demo_course，不得猜测。用一句中文回答，不输出计划。')
    async with asyncio.timeout(240):
        result = await agent.ainvoke({'messages': [{'role': 'user', 'content': '请查询虚构实验的观察窗口是多少分钟。'}]},
                                    config={'recursion_limit': 8})
    answer = result['messages'][-1]
    if not calls or answer.type != 'ai' or answer.tool_calls or '17' not in str(answer.content):
        raise RuntimeError('Local model did not complete the expected tool call and final answer')
    print(json.dumps({'status': 'passed', 'model': model_name, 'tool_calls': len(calls), 'answer': answer.content}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Installed model tag from ollama list')
    args = parser.parse_args()
    asyncio.run(main(args.model))
