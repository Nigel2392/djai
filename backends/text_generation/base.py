from abc import ABC, abstractmethod

class Text:
    def __init__(self, text: str, speaker: str):
        self.speaker = speaker
        self.text = text

    def __hash__(self):
        return hash((self.text, self.speaker))
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.text == other
        
        return self.text == other.text and self.speaker == other.speaker

class BaseTextGenerationBackend(ABC):
    class Conversation(ABC):
        def __init__(self, backend: "BaseTextGenerationBackend", clear: bool=True):
            self.backend = backend
            self._messages: list[Text] = []
            self.clear = clear

        def __enter__(self):
            if self.clear:
                self._messages.clear()
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            pass
        
        @property
        def messages(self) -> list[Text]:
            return self._messages.copy()
        
        def add_message(self, message: Text):
            self._messages.append(message)

        def chat(self, message: Text) -> Text:
            self.add_message(message)
            backend_generated = self.backend.generate_text(self.messages)
            self.add_message(backend_generated)
            return backend_generated

    def __init__(self, options: dict):
        self.options = options

    def create_conversation(self, clear: bool=False) -> Conversation:
        return self.Conversation(self, clear=clear)
    
    @abstractmethod
    def generate_text(self, conversation: list[Text]) -> Text:
        pass

