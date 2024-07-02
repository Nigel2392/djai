Initialize any of the text-classification backends.
Example:

```python
DJAI_BACKENDS = {
    "distilbert-cased-sentiment": {
        "CLASS": "djai.backends.DistilBertCasedSentimentQualifierBackend",
        "OPTIONS": {
            "model_name": "Y:\\models\\distilbert-base-multilingual-cased-sentiments-student",
        }
    },
    "distilbert-cased-toxicity": {
        "CLASS": "djai.backends.DistilBertCasedToxicityQualifierBackend",
        "OPTIONS": {
            "model_name": "Y:\\models\\distilbert-base-dutch-cased-toxic-comments",
        }
    },
    "distilbert-cased-emotion": {
        "CLASS": "djai.backends.DistilBertUncasedEmotionQualifierBackend",
        "OPTIONS": {
            "model_name": "Y:\\models\\distilbert-base-uncased-emotion",
        }
    },
    "bert-base-multilingual-uncased-sentiment": {
        "CLASS": "djai.backends.BaseHFQualifierBackend",
        "OPTIONS": {
            "model_name": "Y:\\models\\bert-base-multilingual-uncased-sentiment",
            "property_map": [],
            "sort_key": (0, False),
            "labels": [
                "5 stars",
                "4 stars",
                "3 stars",
                "2 stars",
                "1 star",
            ],
        },
    },
}
```