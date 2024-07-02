
try:
    from llama_cpp import Llama
    from typing import Callable, Dict
    from . import base

    class LocalLlamaTextGenerationBackend(base.BaseTextGenerationBackend):
        class Conversation(base.BaseTextGenerationBackend.Conversation):
            backend: "LocalLlamaTextGenerationBackend"

        def __init__(self, options: dict):
            super().__init__(options)
            self.llm_kwargs = options.get("llm_kwargs")
            self.call_kwargs = options.get("call_kwargs") or {}
            self.get_text: Callable[[Dict], base.Text] = options.get("get_text")
            self._llm = None
            self.llm = options.get("llm")

            if self.get_text is None:
                raise ValueError("get_text must be provided to LocalLlamaBackend")

        def generate_text(self, conversation: list[base.Text]) -> base.Text:
            messages = []
            for message in conversation:
                messages.append(
                    {"mode": message.speaker, "message": message.text},
                )
            response = self.llm.create_chat_completion(
                messages=messages,
                **self.call_kwargs,
            )

            return self.get_text(response)

        @property
        def llm(self):
            return self._llm

        @llm.setter
        def llm(self, value):
            if isinstance(value, str):
                self._llm = Llama(model_path=value, **self.llm_kwargs)
            else:
                self._llm = value 

except ImportError:
    LocalLlamaTextGenerationBackend = None
