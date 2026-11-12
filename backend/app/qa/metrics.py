"""记录模型服务实际返回的用量，缺失时保留未知状态。"""

from langchain_core.callbacks import AsyncCallbackHandler
from app.services import runs


class RunUsage(AsyncCallbackHandler):
    """累计一轮问答中模型报告的 Token 用量。

    Attributes:
        run_id: 当前问答运行 ID。
        calls: 已开始的模型调用次数。
        reported: 已返回用量信息的调用次数。
        input_tokens: 已报告的输入 Token 总数。
        output_tokens: 已报告的输出 Token 总数。
    """

    def __init__(self, run_id):
        """绑定运行并将模型调用及用量计数归零。

        Args:
            run_id: 问答运行 ID。
        """
        self.run_id = run_id
        self.calls = 0
        self.reported = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def publish(self):
        """保存累计用量和报告完整性，未知 Token 数保留为空值。"""
        runs.patch_trace(
            self.run_id,
            {
                "usage": {
                    "model_calls": self.calls,
                    "calls_with_usage": self.reported,
                    "input_tokens": self.input_tokens if self.reported else None,
                    "output_tokens": self.output_tokens if self.reported else None,
                    "complete": self.reported == self.calls,
                }
            },
        )

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        """累计一次已开始的对话模型调用。

        Args:
            serialized: LangChain 提供的模型序列化信息。
            messages: 发送给模型的各组输入消息。
            kwargs: 调用方附加的关键字参数。
        """
        self.calls += 1
        self.publish()

    async def on_llm_end(self, response, **kwargs):
        """累计模型结果中的用量元数据。

        Args:
            response: 包含候选生成结果的 LangChain 响应。
            kwargs: 调用方附加的关键字参数。
        """
        usage = next(
            (
                getattr(item.message, "usage_metadata", None)
                for group in response.generations
                for item in group
                if getattr(getattr(item, "message", None), "usage_metadata", None)
            ),
            None,
        )
        if usage:
            self.reported += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
        self.publish()
