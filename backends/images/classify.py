try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForImageClassification, AutoProcessor
    from PIL import Image

    from typing import Union
    from pathlib import Path

    from ..utils import ClassifierDict
    from .base import (
        BaseImageClassifierBackend,
    )

    class BaseHFImageClassifierBackend(BaseImageClassifierBackend):
        model_name = None
        processor_class: AutoProcessor = AutoProcessor
        model_class: AutoModelForImageClassification = AutoModelForImageClassification
        label_names = []
        positive_field = None
        negative_field = None

        def __init__(self, options: dict):
            model_name = options.get("model", self.model_name)
            if model_name is None:
                raise ValueError("model_name is not specified")
            
            self.positive_field = options.get("positive_field", self.positive_field)
            self.negative_field = options.get("negative_field", self.negative_field)

            processor_options = options.get("processor_options", {})
            model_options = options.get("model_options", {
                "num_labels": len(self.label_names),
                "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
                "problem_type": "multi_label_classification"
            })

            self.min_positivity = options.get("min_positivity", 80)
            self.min_distance = options.get("min_distance", self.min_positivity / 10)
            self.processor = self.processor_class.from_pretrained(model_name, **processor_options)
            self.model = self.model_class.from_pretrained(model_name, **model_options)
            self.model.eval()

        def predict(self, image: Union[Image.Image, str, Path, bytes], formats: list[str] = None, min_positivity: int = None, min_distance: int = None, **mdl_kwargs) -> ClassifierDict[str, float]:

            if not isinstance(image, Image.Image):
                image = Image.open(image, formats=formats)

            with torch.inference_mode():
                inputs = self.processor(images=image, return_tensors="pt")
                inputs.to(self.model.device)
                outputs = self.model(**inputs, **mdl_kwargs)

            # Apply sigmoid to logits
            probs = F.sigmoid(outputs.logits).squeeze()

            sentiment_analysis_results = []
            for label_id, prob in enumerate(probs):
                sentiment_analysis_results.append((
                    self.model.config.id2label[label_id],
                    round(prob.item() * 100, 3)
                ))

            data = ClassifierDict(
                sentiment_analysis_results,
                pos_field=self.positive_field,
                neg_field=self.negative_field,
                positivity=min_positivity or self.min_positivity,
                distance=min_distance or self.min_distance,
                attrs={
                    "image": image,
                    "original_outputs": probs,
                    "classifier": self,
                }
            )

            return data

    class NSFWImageClassifierBackend(BaseHFImageClassifierBackend):
        model_name = "Falconsai/nsfw_image_detection"
        processor_class: AutoProcessor = AutoProcessor
        model_class: AutoModelForImageClassification = AutoModelForImageClassification
        label_names = ["normal", "nsfw"]
        positive_field = "normal"
        negative_field = "nsfw"

except ImportError:
    BaseHFImageClassifierBackend = None
    NSFWImageClassifierBackend = None