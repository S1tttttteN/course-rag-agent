"""提供可记录输入的脚本化聊天模型和流式测试替身。"""

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


def call(name, args, identifier='call1'):
    """构造包含一次工具调用的模型消息。

    Args:
        name: 工具名称。
        args: 传给工具的参数字典。
        identifier: 工具调用 ID。

    Returns:
        带工具调用记录的 AI 消息。
    """
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': identifier}])


class ScriptedModel(BaseChatModel):
    """按预设顺序返回消息的测试对话模型。

    Attributes:
        responses: 预设工具调用或回答消息。
        cursor: 下次生成使用的响应位置。
        seen: 已记录的输入消息列表。
        wait_after: 达到该响应位置后触发异步等待。
    """

    responses: list = Field(default_factory=list)
    cursor: int = 0
    seen: list = Field(default_factory=list)
    wait_after: int | None = None

    @property
    def _llm_type(self):
        """提供用于识别测试模型的类型标识。

        Returns:
            固定的模型类型字符串。
        """
        return 'scripted-test-model'

    def bind_tools(self, tools, **kwargs):
        """接受框架提供的工具绑定。

        Args:
            tools: 框架传入的工具列表。
            kwargs: 调用方附加的关键字参数。

        Returns:
            当前脚本模型。
        """
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """生成下一条预设响应。

        Args:
            messages: 传给模型的消息。
            stop: 框架传入的可选停止词。
            run_manager: 框架传入的可选回调管理器。
            kwargs: 调用方附加的关键字参数。

        Returns:
            包含一条模型响应的生成结果。
        """
        self.seen.append(messages)
        message = self.responses[min(self.cursor, len(self.responses) - 1)].model_copy(deep=True)
        message.id = None
        for tool_call in message.tool_calls:
            tool_call['id'] = f"{tool_call['id']}-{self.cursor}"
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """在指定响应位置模拟等待，再执行预设生成。

        Args:
            messages: 传给模型的消息。
            stop: 框架传入的可选停止词。
            run_manager: 框架传入的可选异步回调管理器。
            kwargs: 调用方附加的关键字参数。

        Returns:
            包含一条模型响应的生成结果。
        """
        if self.wait_after is not None and self.cursor >= self.wait_after:
            await asyncio.sleep(30)
        return self._generate(messages)


class StreamingModel(ScriptedModel):
    """将预设响应转换为测试用流式模型输出。"""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """将预设响应作为工具调用或正文流发送。

        Args:
            messages: 传给模型的消息。
            stop: 框架提供的可选停止词。
            run_manager: 框架提供的可选异步回调管理器。
            kwargs: 调用方附加的关键字参数。

        Yields:
            工具调用块或分段的正文块。
        """
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        if message.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content='', tool_call_chunks=[
                {'name': item['name'], 'args': json.dumps(item['args']), 'id': item['id'], 'index': index}
                for index, item in enumerate(message.tool_calls)]))
        else:
            for position in range(0, len(message.content), 5):
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content=message.content[position:position + 5],
                    response_metadata=message.response_metadata if position + 5 >= len(message.content) else {},
                ))
                await asyncio.sleep(.01)


class PausedModel(StreamingModel):
    """模拟发送首段正文后停顿的模型。"""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """发出一个正文片段后等待，用于测试运行取消。

        Args:
            messages: 流式接口收到的输入消息。
            stop: 框架提供的可选停止词。
            run_manager: 框架提供的可选异步回调管理器。
            kwargs: 调用方附加的关键字参数。

        Yields:
            等待前发出的正文片段。
        """
        yield ChatGenerationChunk(message=AIMessageChunk(content='已生成的自然回答。'))
        await asyncio.sleep(30)


class FailingModel(StreamingModel):
    """模拟发布部分正文后失败的模型。"""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """发出正文片段后触发模拟服务商错误。

        Args:
            messages: 流式接口收到的输入消息。
            stop: 框架提供的可选停止词。
            run_manager: 框架提供的可选异步回调管理器。
            kwargs: 调用方附加的关键字参数。

        Yields:
            失败前发出的正文片段。

        Raises:
            RuntimeError: 用于验证服务商错误处理的模拟异常。
        """
        yield ChatGenerationChunk(message=AIMessageChunk(content='已生成的部分。'))
        raise RuntimeError('private-provider-error-fixture')
