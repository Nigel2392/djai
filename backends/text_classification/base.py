from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
from ..utils import AttrDict

class BaseTextQualifierBackend(ABC):
    def __init__(self, options: dict[str, Any]):
        self.options = options

    @abstractmethod
    def qualify(self, text: str, **kwargs) -> "AttrDict[str, Any]":
        raise NotImplementedError("qualify must be implemented by subclasses of BaseTextQualifierBackend")

class BaseZeroShotClassifierBackend(ABC):
    def __init__(self, options: dict[str, Any]):
        self.options = options

    @abstractmethod
    def classify(self, prompt: str, labels: list, hypothesis: str = None, **kwargs) -> AttrDict[str, float]:
        pass

    @abstractmethod
    def premise(self, prompt: str, hypothesis: str) -> AttrDict[str, float]:
        pass

