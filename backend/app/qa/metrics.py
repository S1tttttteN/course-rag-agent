"""记录模型服务实际返回的用量，缺失时保留未知状态。"""

from langchain_core.callbacks import AsyncCallbackHandler
from app.services import runs


class RunUsage(AsyncCallbackHandler):
    """累计一轮问答中模型报告的 Token 用量。

    Attributes:
        run_id: 当前问答运行 ID，离线评测时为空。
        calls: 已开始的模型调用次数。
        summary_calls: 已开始的摘要模型调用次数。
        reported: 已返回用量信息的调用次数。
        input_tokens: 已报告的输入 Token 总数。
        output_tokens: 已报告的输出 Token 总数。
    """

    def __init__(self, run_id=None):
        """绑定运行并将模型调用及用量计数归零。

        Args:
            run_id: 问答运行 ID，省略时只收集用量。
        """
        self.run_id = run_id
        self.calls = 0
        self.summary_calls = 0
        self.reported = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def summary(self):
        """返回累计用量及报告完整性，未知 Token 数使用空值。"""
        return {
            "model_calls": self.calls,
            "summary_calls": self.summary_calls,
            "calls_with_usage": self.reported,
            "input_tokens": self.input_tokens if self.reported else None,
            "output_tokens": self.output_tokens if self.reported else None,
            "complete": self.reported == self.calls,
        }

    def publish(self):
        """将可关联运行的模型用量保存到执行记录。"""
        if self.run_id is not None:
            runs.patch_trace(self.run_id, {"usage": self.summary()})

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        """累计一次已开始的对话模型调用。

        Args:
            serialized: LangChain 提供的模型序列化信息。
            messages: 发送给模型的各组输入消息。
            **kwargs: LangChain 回调参数，包含内部调用的来源标记。
            kwargs: 调用方附加的关键字参数。
        """
        self.calls += 1
        if (kwargs.get("metadata") or {}).get("lc_source") == "summarization":
            self.summary_calls += 1
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
