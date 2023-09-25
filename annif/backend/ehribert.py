import json
import os
from os.path import dirname
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from . import transformer


class EhriBertBackend(transformer.BaseTransformerBackend):
    """Recogniser fine-tuned using bert-base-multilingual-cased."""
    name = "ehribert"
    base_uri = 'http://data.ehri-project.eu/vocabularies/ehri-terms/'
    initialized = False
    is_trained = True
    subject_id = 0
    modification_time = None
    _label_texts = None # internal multilingual EHRI label, e.g. ehri_terms-123 to English label
    _label_uris = None # mapping of internal label to URI
    _uri_to_label = {}
    _label_to_uri = {}
    _config = None
    _tokenizer = None
    _model = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        model_path = os.path.join(dirname(dirname(dirname(__file__))),  "models", "finetuned-bert-base-multilingual-cased-ehri-terms")
        for uri in self._terms.keys():
            tid = uri.replace(self.base_uri, 'ehri_terms-')
            self._uri_to_label[uri] = tid
            self._label_to_uri[tid] = uri

        with open(os.path.join(model_path, "config.json"), "r") as c:
            config = json.load(c)
            self._label_texts = {}
            # Cull labels to those in the config (which is inconvenient...)
            for uri, text in self._terms.items():
                tid = self._uri_to_label[uri]
                if tid in config["label2id"]:
                    self._label_texts[tid] = text
            self._label_uris = {}
            for tid in self._label_texts.keys():
                uri = self._label_to_uri[tid]
                self._label_uris[tid] = uri

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self.initialized = True

    def run_pipeline(self, texts: list[str], params: dict[str, Any] = None, num: int = 10):
        results = []

        threshold = float(params.get("threshold", 0.0))

        for text in texts:
            encoding = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encoding = {k: v.to(self._model.device) for k,v in encoding.items()}

            outputs = self._model(**encoding)
            logits = outputs.logits

            id2label = self._config.id2label
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.squeeze().cpu())
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= threshold)] = 1

            sorted_predictions = []

            for idx, label in enumerate(predictions):
                if label == 1.0:
                    actual_score = probs[idx].item()
                    predicted_label = id2label[idx]
                    pref_label = self._label_texts[predicted_label]
                    uri = self._label_to_uri[predicted_label]
                    sorted_predictions.append((uri, pref_label, actual_score))

            sorted_predictions = sorted(sorted_predictions, key=lambda x: x[2], reverse=True)[:num]
            results.append(sorted_predictions)
        return results

