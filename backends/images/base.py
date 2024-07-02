from abc import ABC, abstractmethod
from PIL import Image
from typing import Union, Any
from pathlib import Path
from ..utils import ClassifierDict

class BaseImageBackend(ABC):
    def __init__(self, options: dict):
        self.options = options

    @abstractmethod
    def caption(self, fp: Union[Image.Image, str, Path, bytes], formats: Union[list, tuple]=None) -> list[str]:
        pass


class BaseImageClassifierBackend(ABC):
    def __init__(self, options: dict):
        self.options = options

    @abstractmethod
    def predict(self, image: Union[Image.Image, str, Path, bytes], formats: list[str] = None, min_positivity: int = None, min_distance: int = None, **mdl_kwargs) -> ClassifierDict[str, float]:
        pass
