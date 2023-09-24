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
    initialized = False
    is_trained = True
    subject_id = 0
    modification_time = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        from transformers import pipeline
        model = os.path.join(dirname(dirname(dirname(__file__))),  "models", "finetuned-bert-base-multilingual-cased-ehri-terms")

        from torch import cuda
        device = 0 if cuda.is_available() else -1
        self._zs = pipeline("zero-shot-classification", model=model, device=device)

        self.initialized = True
