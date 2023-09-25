import json
import os
from typing import Union, Any

from . import backend
from . import transformer
from annif.corpus.types import DocumentCorpus
from annif.suggestion import SubjectSuggestion
from os.path import dirname

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

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        from transformers import pipeline
        model = os.path.join(dirname(dirname(dirname(__file__))),  "models", "finetuned-bert-base-multilingual-cased-ehri-terms")
        with open(os.path.join(model, "config.json"), "r") as c:
            config = json.load(c)
            self._label_texts = {}
            # Cull labels to those in the config (which is inconvenient...)
            for uri, text in self._terms.items():
                tid = uri.replace(self.base_uri, 'ehri_terms-')
                if tid in config["label2id"]:
                    self._label_texts[tid] = text
            self._label_uris = {}
            for tid in self._label_texts.keys():
                uri = tid.replace('ehri_terms-', self.base_uri)
                self._label_uris[tid] = uri

        from torch import cuda
        device = 0 if cuda.is_available() else -1
        self._zs = pipeline("zero-shot-classification", model=model, device=device)

        self.initialized = True

    def run_pipeline(self, texts: list[str], num: int = 10) -> list[Union[tuple[str, str, float], None]]:  # (concept_id, concept_text, confidence)
        """Recognise a term from the vocabulary in the given text
        :param text: Text to recognise
        :return: (concept_id, concept_text, confidence) or None if no match
        """
        if len(texts) == 0:
            return []

        results = self._zs(texts, candidate_labels=list(self._label_texts.keys()), multi_label=num > 1)
        def suggestions(result):
            num_results = len(result["labels"])
            return [(self._label_uris[result["labels"][i]], self._label_texts[result["labels"][i]], result["scores"][i]) for i in range(min(num, num_results))]

        return [suggestions(r) for r in results]

