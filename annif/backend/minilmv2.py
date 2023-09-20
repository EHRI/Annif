from typing import Union, Any

from . import backend
from . import transformer
from annif.suggestion import SubjectSuggestion, SuggestionBatch


class MiniLmV2Backend(transformer.BaseTransformerBackend):
    """Recogniser using the MiniLMv2 model from Huggingface Transformers"""
    name = "minilmv2"
    initialized = False
    is_trained = True
    modification_time = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        from transformers import pipeline
        model = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

        from torch import cuda
        device = 0 if cuda.is_available() else -1
        self._zs = pipeline("zero-shot-classification", model=model, device=device)

        self.initialized = True

