from typing import Union, Any

from . import backend
from . import transformer
from annif.corpus.types import DocumentCorpus
from annif.suggestion import SubjectSuggestion


class XlmRobertaBackend(transformer.BaseTransformerBackend):
    """Recogniser using the joeddav/xlm-roberta-large-xnli from HuggingFace."""

    name = "xlmroberta"
    initialized = False
    is_trained = True
    subject_id = 0
    modification_time = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        from transformers import pipeline

        model = "joeddav/xlm-roberta-large-xnli"

        from torch import cuda

        device = 0 if cuda.is_available() else -1
        self._zs = pipeline(
            "zero-shot-classification", model=model, device=device, use_auth_token=True
        )

        self.initialized = True
