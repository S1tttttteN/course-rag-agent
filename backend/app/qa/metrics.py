"""Provider-reported usage only. Unknown usage is never estimated as zero."""
from langchain_core.callbacks import AsyncCallbackHandler
from app.services import runs


class RunUsage(AsyncCallbackHandler):
    def __init__(self, run_id):
        self.run_id = run_id
        self.calls = 0
        self.reported = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def publish(self):
        runs.patch_trace(self.run_id, {"usage": {
            "model_calls": self.calls, "calls_with_usage": self.reported,
            "input_tokens": self.input_tokens if self.reported else None,
            "output_tokens": self.output_tokens if self.reported else None,
            "complete": self.reported == self.calls,
        }})

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        self.calls += 1
        self.publish()

    async def on_llm_end(self, response, **kwargs):
        usage = next((getattr(item.message, "usage_metadata", None)
                      for group in response.generations for item in group
                      if getattr(getattr(item, "message", None), "usage_metadata", None)), None)
        if usage:
            self.reported += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
        self.publish()
