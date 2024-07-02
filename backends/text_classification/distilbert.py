from typing import Any, List, Tuple, Union

from . import base
from .. import utils

# Minimum percentage of confidence to be considered positive
_positivity_min = 60
# Minimum distance between the value and every other value to be considered positive
_distance_min = 5
    

class BaseQualifierDict(utils.AttrDict):
    # Original text
    original: str
    # Locale of the text (Used when splitting the text into sentences for processing)
    locale: str
    # List of raw results gained from the split sentences.
    results: List[Tuple[str, float]]

    def __str__(self):
        return self.original
        
    
class DistilBertCasedSentimentQualifierBackend(BaseQualifierDict):
    # Take a guess if the sentiment is <sentiment>
    is_positive: bool
    is_negative: bool
    is_neutral: bool

    # Confidence of the sentiment
    positive: float
    negative: float
    neutral: float

class DistilBertCasedToxicityQualifierBackend(BaseQualifierDict):
    # Take a guess if the sentiment is <sentiment>
    is_toxic: bool
    is_non_toxic: bool

    # Confidence of the toxicity
    toxic: float
    non_toxic: float

class DistilBertUncasedEmotionQualifierBackend(BaseQualifierDict):
    # Take a guess if the sentiment is <sentiment>
    is_love: bool
    is_joy: bool
    is_sadness: bool
    is_anger: bool
    is_surprise: bool
    is_fear: bool

    # Confidence of the emotion
    love: float
    joy: float
    sadness: float
    anger: float
    surprise: float
    fear: float

# QualifierDict = Union[
#     DistilBertCasedSentimentQualifierBackend,
#     DistilBertCasedToxicityQualifierBackend,
#     DistilBertUncasedEmotionQualifierBackend
# ]
    
from django.db.models import JSONField
import json

class _QualifierJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "_json"):
            return obj._json()
        return super().default(obj)

class QualifierDict:
    def __init__(self, 
                 backend:       "BaseHFQualifierBackend"  = None, 
                 labels:        list[str]               = None, 
                 data:          dict[str, float]        = None, 
                 original:      str                     = None, 
                 locale:        str                     = None, 
                 results:       list[tuple[str, float]] = None, 
                 property_map:  dict[str, callable]     = None,
                 sort_key:      callable                = None
        ):
        self.backend:       "BaseHFQualifierBackend"  = backend
        self.labels:        list[str]               = labels or []
        self.data:          dict[str, float]        = data or {}
        self.original:      str                     = original
        self.locale:        str                     = locale
        self.property_map:  dict[str, callable]     = property_map or {}
        self.results:       list[tuple[str, float]] = results or []
        self._sort_key:     callable                = sort_key\
            if (sort_key is not None) else lambda x: x[1]
        self._mutable                               = True

    def __str__(self):
        data = self.data.items()
        sort_key = self._sort_key
        reverse = True

        if not callable(sort_key):
            if isinstance(sort_key, int):
                k = sort_key
                sort_key = lambda x: x[k]
            elif isinstance(sort_key, str):
                k = sort_key
                if k.startswith("-"):
                    k = int(k[1:])
                    reverse = False
                else:
                    k = int(k)
                sort_key = lambda x: x[k]
            elif isinstance(sort_key, (list, tuple)):
                k, reverse = sort_key
                sort_key = lambda x: x[k]

        data = sorted(data, key=sort_key, reverse=reverse)
        data = [f"{k.upper()}: {v:.1f}" for k, v in data if v > 10]
        return ' / '.join(data)
    
    def __getattr__(self, attr):
        if attr in self.property_map:
            prop, func = self.property_map[attr]
            return func(self.data, self.data[prop], prop)

        elif attr in self.data:
            return self.data[attr]

        return super().__getattr__(attr)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        if not self._mutable:
            raise AttributeError("Cannot set attribute on immutable AttrDict")
        self.data[key] = value

    def update(self, other):
        self.data.update(other)

    def deconstruct(self):
        return (
            f"djai.backends.{self.__class__.__name__}",
            [],
            {
                "backend": self.backend,
                "labels": self.labels,
                "data": self.data,
                "original": self.original,
                "locale": self.locale,
                "results": self.results,
                "property_map": self.property_map,
                "sort_key": self._sort_key,
            },
        )

    def _json(self):
        return {
            "backend": self.backend,
            "original": self.original,
            "locale": self.locale,
            "results": self.results,
            "data": self.data,
        }
    
_get_backend = None

class QualifierDictField(JSONField):
    def __init__(self, *args: Any, backend: Union["BaseHFQualifierBackend", str] = None, preload: bool = False, **kwargs: Any) -> None:
        kwargs["default"] = dict
        kwargs["encoder"] = _QualifierJSONEncoder
        super().__init__(*args, **kwargs)
        self._backend = backend
        if preload:
            self._backend = self.backend

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs["backend"] = self._backend
        return name, path, args, kwargs

    @property
    def backend(self) -> "BaseHFQualifierBackend":
        if isinstance(self._backend, str):
            global _get_backend
            if _get_backend is None:
                from djai.backends import get_backend
                _get_backend = get_backend
            self._backend = _get_backend(self._backend)
        return self._backend
    
    def qualify(self, text: str, locale: str = None) -> QualifierDict:
        return self.backend.qualify(text, locale)

    def to_python(self, value):
        value = super().to_python(value)
        if isinstance(value, dict):
            return self.backend.from_dict(**value)
        return value
    
    def from_db_value(self, value, expression, connection):
        value = super().from_db_value(value, expression, connection)
        if value is None:
            return value
        return self.backend.from_dict(**value)

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizerFast, DistilBertForSequenceClassification
    from typing import Any, Callable, Tuple
    from django.conf import settings

    import math, os, threading, copy

    from ..utils import (
        languages_nltk_full,
    )

    from nltk.tokenize import sent_tokenize, LineTokenizer

    class BaseHFQualifierBackend(base.BaseTextQualifierBackend):
        model_name = None
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForSequenceClassification
        value_class: QualifierDict = QualifierDict
        property_map: dict[str, Tuple[str, Callable[["QualifierDict", Any, str], Any]]] = None
        sort_key: Callable[[Tuple[str, float]], float] = None
        labels = None

        def __init__(self, options: dict):
            super().__init__(options)
            model_path = options.get("model_name", self.model_name)

            if self.__class__ is BaseHFQualifierBackend:
                self.model_class = options.get("model_class", self.model_class)
                self.tokenizer_class = options.get("tokenizer_class", self.tokenizer_class)
                self.value_class = options.get("value_class", self.value_class)
                self.property_map = options.get("property_map", self.property_map)
                self.sort_key = options.get("sort_key", self.sort_key)
                self.labels = options.get("labels", self.labels)

            tokenizer_options = options.get("tokenizer_options", {})
            model_options = options.get("model_options", {
                "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
            })

            self.tokenizer = self.tokenizer_class.from_pretrained(model_path, **model_options)
            self.model = self.model_class.from_pretrained(model_path, problem_type="multi_label_classification", **tokenizer_options)
            self.thread_count = options.get("thread_count", (os.cpu_count() or 2) - 1)
            self.model.eval()

        def preprocess_text(self, text: str, locale: str = None) -> str:
            """
                Preprocess the text before it is sent to the model.
                This is useful for removing unwanted characters, adding special tokens, translations, etc.
            """
            return text
        
        def preprocess_batch(self, batch: list[str], locale: str) -> list[str]:
            """
                Preprocess the batch before it is sent to the model.
                This is useful for removing unwanted characters, adding special tokens, translations, etc.
            """
            return batch
        
        def preprocess_paragraphs(self, paragraphs: list[str], locale: str = None) -> list[str]:
            """
                Preprocess the paragraphs before they are sent to the model.
                This is useful for removing unwanted characters, adding special tokens, translations, etc.
                Beware! A single paragraph might still be a very long string.
            """
            return paragraphs
        
        def from_dict(self, data: dict, original: str = None, locale: str = None, results: list[tuple[str, float]] = None, backend: str = None) -> QualifierDict:
            """
                Create a new QualifierDict object from a dictionary.
                Backend is unused.
            """
            new: QualifierDict = self.value_class(
                backend=self,
                data=data,
                labels=self.labels,
                property_map=self.property_map,
                original=original,
                locale=locale,
                results=results,
                sort_key=self.sort_key,
            )
            new._mutable = False
            return new
        
        def _qualify(self, text: str) -> dict[str, float]:
            text = self.preprocess_text(text)
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs.to(self.model.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Apply sigmoid to logits
            probs = F.sigmoid(logits).squeeze()

            results = [None] * len(probs)
            for index, prob in enumerate(probs):
                label_str = self.model.config.id2label[index]
                probability = prob.item() * 100
                results[index] = (label_str, probability)

            return dict(results)
        
        def _thread_qualify(self, batched_sentences: list[str], index, results: list[dict[str, float]], locale: str):
            batched_sentences = self.preprocess_batch(batched_sentences, locale)
            for k, sentence in enumerate(batched_sentences):
                results[index + k] = self._qualify(sentence, locale)

        def qualify(self, text: str, locale: str=settings.LANGUAGE_CODE) -> QualifierDict:
            # Initialize the class instances here to reduce strain on the garbage collector...
            ln_tknzr = LineTokenizer()


            # Split the text into paragraphs.
            paragraphs = ln_tknzr.tokenize(text)
            paragraphs = self.preprocess_paragraphs(paragraphs, locale)

            # NLTK supports limited languages to split sentences.
            # If the language is not supported, english is used by default.
            default_lang_code = settings.LANGUAGE_CODE.split("-")[0].lower()
            language = languages_nltk_full.get(
                locale.split("-")[0].lower(), 
                languages_nltk_full[default_lang_code], # Default to settings.LANGUAGE_CODE
            )
            # Split the paragraphs into sentences.
            split_paragraphs: list[list[str]] = [sent_tokenize(paragraph, language=language) for paragraph in paragraphs]
            # Pre-initialize the list to reduce strain on the garbage collector...
            sentiment_analysis_results: list[dict[str, float]] = [None] * (
                len(paragraphs) * sum(map(len, split_paragraphs))
            )
            for idx, sentences in enumerate(split_paragraphs):
                # Batch the sentences to reduce the number of calls to the model, 
                # and speed up the translation.
                batches = math.ceil(len(sentences) / self.thread_count)
                threads = [None] * batches
                for k_idx in range(batches):
                    # Create batches of sentences to translate.
                    # Format the batches according to what the translator needs.
                    batched_sentences = sentences[k_idx*self.thread_count:(k_idx+1)*self.thread_count]
                    # batch.append(sentence_batch)
                
                    # Yield a batch of size `batch_size` (or less) to the caller.
                    threads[k_idx] = threading.Thread(target=self._thread_qualify, args=(
                        batched_sentences, 
                        (idx * self.thread_count) + (k_idx * self.thread_count), 
                        sentiment_analysis_results,
                        locale,
                    ))
                    threads[k_idx].start()

                for thread in threads:
                    thread.join()
                    
            d: dict[str, float] = {label: 0 for label in self.model.config.id2label.values()}
            for result in sentiment_analysis_results:
                for label_str, probability in result.items():
                    d[label_str] += probability
                    
            for label_str in d:
                d[label_str] = d[label_str] / len(sentiment_analysis_results)

            return self.from_dict(
                d,
                original=text,
                locale=locale,
                results=sentiment_analysis_results,
            )

        def deconstruct(self):
            return (
                f"djai.backends.{self.__class__.__name__}",
                [],
                {
                    "options": self.options,
                },
            )
        
        def _json(self):
            options = copy.deepcopy(self.options)
            if "sort_key" in options:
                del options["sort_key"]
            if "property_map" in options:
                del options["property_map"]
            return {
                "model": self.model_name,
                "options": options,
            }
        
    class DistilBertCasedSentimentQualifierBackend(BaseHFQualifierBackend):
        model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        model_class = DistilBertForSequenceClassification
        tokenizer_class = DistilBertTokenizerFast
        labels = ["positive", "negative", "neutral"]
        property_map = {
            "is_positive": ("positive", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_negative": ("negative", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_neutral": ("neutral", utils.attr_is_positive(_positivity_min, _distance_min)),
        }
        
    class DistilBertCasedToxicityQualifierBackend(BaseHFQualifierBackend):
        model_name = "ml6team/distilbert-base-dutch-cased-toxic-comments"
        model_class = DistilBertForSequenceClassification
        tokenizer_class = DistilBertTokenizerFast
        labels = ["toxic", "non-toxic"]
        property_map = {
            "is_toxic": ("toxic", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_non_toxic": ("non-toxic", utils.attr_is_positive(_positivity_min, _distance_min)),
        }

    class DistilBertUncasedEmotionQualifierBackend(BaseHFQualifierBackend):
        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        model_class = DistilBertForSequenceClassification
        tokenizer_class = DistilBertTokenizerFast
        labels = ["love", "joy", "sadness", "anger", "surprise", "fear"]
        property_map = {
            "is_love": ("love", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_joy": ("joy", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_sadness": ("sadness", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_anger": ("anger", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_surprise": ("surprise", utils.attr_is_positive(_positivity_min, _distance_min)),
            "is_fear": ("fear", utils.attr_is_positive(_positivity_min, _distance_min)),
        }

except ImportError as e:
    BaseHFQualifierBackend = None
    DistilBertCasedSentimentQualifierBackend = None
    DistilBertCasedToxicityQualifierBackend = None
    DistilBertUncasedEmotionQualifierBackend = None


# class ContactModel(models.Model):
#     message = models.TextField(
#         verbose_name=_("Message"),
#         help_text=_("Your message."),
#         max_length=1024,
#         blank=False,
#         null=False,
#     )
# 
# 
#     rating = QualifierDictField(
#         # backend="distilbert-cased-toxicity",
#         backend="distilbert-cased-sentiment",
#         verbose_name=_("Rating"),
#         help_text=_("Your rating."),
#         blank=True,
#         null=True,
#         preload=True,
#         default=dict
#     )
# 
#     class Meta:
#         verbose_name = _("Contact Request")
#         verbose_name_plural = _("Contact Requests")
# 
# 
#     @property
#     def rating_field(self) -> QualifierDictField:
#         field: QualifierDictField = self._meta.get_field("rating")
#         return field
# 
#     def save(self, *args, **kwargs):
#         if not self.rating or self.rating.original != self.message:
#             if self.rating_field:
#                 self.rating = self.rating_field.qualify(
#                     self.message, translation.get_language()
#                 )
#         super().save(*args, **kwargs)


