from typing import (
    Iterable,
    Any,
    Callable,
    Tuple,
)
from django.conf import settings
import math

languages_nltk_full = {
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'et': 'estonian',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'it': 'italian',
    'ml': 'malayalam',
    'no': 'norwegian',
    'pl': 'polish',
    'pt': 'portuguese',
    'ru': 'russian',
    'sl': 'slovene',
    'es': 'spanish',
    'sv': 'swedish',
    'tr': 'turkish',
    **getattr(settings, "NLTK_LANGUAGES_DICT", {}),
}

try:
    from nltk.tokenize import LineTokenizer, sent_tokenize

    def text_to_sentences(locale: str, text: str, batch_size: int) -> Iterable[list[str]]:
        # Initialize the class instances here to reduce strain on the garbage collector...
        ln_tknzr = LineTokenizer()

        # Split the text into paragraphs.
        paragraphs = ln_tknzr.tokenize(text)

        # NLTK supports limited languages to split sentences.
        # If the language is not supported, english is used by default.
        default_lang_code = settings.LANGUAGE_CODE.split("-")[0].lower()
        language = languages_nltk_full.get(
            locale.split("-")[0].lower(), 
            languages_nltk_full[default_lang_code], # Default to settings.LANGUAGE_CODE
        )

        for paragraph in paragraphs:
            # Split the paragraph into sentence batches.
            # Batch the sentences to reduce the number of calls to the model, 
            # and speed up the translation.
            sentences = sent_tokenize(paragraph, language=language)
            batches = math.ceil(len(sentences) / batch_size)
            for i in range(batches):
                # Create batches of sentences to translate.
                # Format the batches according to what the translator needs.
                sentence_batch = sentences[i*batch_size:(i+1)*batch_size]
                # batch.append(sentence_batch)

                # Yield a batch of size `batch_size` (or less) to the caller.
                yield sentence_batch
except ImportError:
    def text_to_sentences(locale: str, text: str, batch_size: int) -> Iterable[list[str]]:
        raise NotImplementedError("NLTK is not installed. Please install NLTK to use this feature.")

class ClassifierDict(dict):
    def __init__(self, *args, pos_field: str = None, neg_field: str = None, positivity: int = 80, distance: int = 10, attrs: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_field = pos_field
        self.neg_field = neg_field
        self.positivity = positivity
        self.distance = distance
        self.__attributes = attrs or {}
        self.__accessible = False

    @property
    def is_positive(self):
        if self.pos_field is None or self.neg_field is None:
            raise ValueError("pos_field and neg_field must be set to use is_positive")
        
        pos = self[self.pos_field]
        return pos > self.positivity and\
               (pos - self.distance) > self[self.neg_field]

    @property
    def is_negative(self):
        if self.pos_field is None or self.neg_field is None:
            raise ValueError("pos_field and neg_field must be set to use is_negative")
        
        neg = self[self.neg_field]
        return neg > self.positivity and\
               (neg - self.distance) > self[self.pos_field]
    
    @property
    def attributes(self):
        if not self.__accessible:
            raise ValueError(f"Cannot access attributes outside of context manager (with {self.__class__.__name__}(...) as ...)")
        
        return self.__attributes
    
    def __getattr__(self, attr):
        if attr in self.__attributes:
            if not self.__accessible:
                raise ValueError(f"Cannot access attributes outside of context manager (with {self.__class__.__name__}(...) as ...)")
            return self.__attributes[attr]
        
        if attr in self:
            return self[attr]
        raise AttributeError(attr)

    def __enter__(self):
        self.__accessible = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__accessible = False
        return False


def _set_immutable_attr(self: "AttrDict", attr, value):
    if not self._mutable:
        raise AttributeError("Cannot set attribute on immutable AttrDict")
    super(AttrDict, self).__setattr__(attr, value)


def attr_is_positive(min_positivity: float, min_distance: float):

    """
        Create a function that checks if an attribute is positive.
        The function will check if the attribute is greater than the other attributes,
        and if the attribute is greater than the minimum positivity.
        It will also check if the attribute is greater than the other attributes by a certain distance.
    """

    __min_distance = min_positivity
    __positivity = min_distance

    def _check(d: dict, v: float, chk: str):

        if isinstance(v, bool):
            return v
        others = [d[k] for k in d if k != chk]

        return (
            all([v-__min_distance > o for o in others]) and \
            v > __positivity
        )
    
    return _check


class AttrDict(dict):
    def __init__(self, *args, property_map: dict[str, Tuple[str, Callable[["AttrDict", float, str], Any]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.property_map = property_map or {}
        self._mutable = True
        self.__setattr__ = _set_immutable_attr

    def __getattr__(self, attr):

        if "-" in attr:
            attr = attr.replace("-", "_")

        if attr in self.property_map:
            chk, fn = self.property_map[attr]
            v = self[chk]
            return fn(self, v, chk)
        try:
            return self[attr]
        except KeyError as e:
            raise AttributeError(attr) from e

