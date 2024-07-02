try:

    from typing import Tuple
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch

    from .. import utils
    from .base import BaseZeroShotClassifierBackend

    class BaseHFZeroShotClassifierBackend(BaseZeroShotClassifierBackend):
        model_name = None
        tokenizer_class: AutoTokenizer = AutoTokenizer
        model_class: AutoModelForSequenceClassification = AutoModelForSequenceClassification
        label_names = ["entailment", "neutral", "contradiction"]

        def __init__(self, options: dict):
            model_name = options.get("model", self.model_name)
            if model_name is None:
                raise ValueError("model_name is not specified")

            tokenizer_options = options.get("tokenizer_options", {})
            model_options = options.get("model_options", {})
            self.tokenizer = self.tokenizer_class.from_pretrained(model_name, **tokenizer_options)
            self.model = self.model_class.from_pretrained(model_name, problem_type="multi_label_classification", **model_options)
            self.model.eval()
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.model.device,  
            )

        def to_dict(self, labels: list[str], values: list[float], data: list[Tuple[str, float]] = None) -> utils.AttrDict[str, float]:

            if data:
                prediction = data
            else:
                prediction = zip(labels, values)

            property_map = {}
            for label in labels:
                property_map[f"is_{label}"] = (label, utils.attr_is_positive(80, 10))

            d = utils.AttrDict(prediction, property_map=property_map)
            d._mutable = False
            return d

        def classify(self, prompt: str, labels: list, hypothesis: str = None, **kwargs) -> utils.AttrDict[str, float]:
            if hypothesis is not None:
                kwargs["hypothesis_template"] = hypothesis

            # Classify the prompt with the labels (hypothesis is optional)
            with torch.inference_mode():
                data = self.classifier(prompt, labels, **kwargs)

            return self.to_dict(
                data["labels"],
                # Convert the score to a percentage (0-100)
                map(lambda x: round(x * 100, 3), data["scores"]),
            )

        def premise(self, prompt: str, hypothesis: str) -> utils.AttrDict[str, float]:
            with torch.inference_mode():
                inputs = self.tokenizer(prompt, hypothesis, truncation=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                output = self.model(**inputs)

            logits = output["logits"][0]
            prediction = torch.softmax(logits, -1).tolist()

            return self.to_dict(
                self.label_names,
                # Convert the score to a percentage (0-100)
                map(lambda x: round(x * 100, 3), prediction),
            )


    class DebertaZeroShotClassifierBackend(BaseHFZeroShotClassifierBackend):
        model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
        label_names = ["entailment", "neutral"]


    class MDebertaZeroShotClassifierBackend(BaseHFZeroShotClassifierBackend):
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        label_names = ["entailment", "neutral", "contradiction"]

except ImportError:
    BaseHFZeroShotClassifierBackend = None
    DebertaZeroShotClassifierBackend = None
    MDebertaZeroShotClassifierBackend = None
