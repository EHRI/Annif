from . import transformer


class MDeBertaBackend(transformer.BaseTransformerBackend):
    """Classifier using MoritzLaurer/mDeBERTa-v3-base-mnli-xnli from HuggingFace."""

    name = "mdeberta"
    initialized = False
    is_trained = True
    subject_id = 0
    modification_time = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        from transformers import pipeline

        model = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

        from torch import cuda

        device = 0 if cuda.is_available() else -1
        self._zs = pipeline(
            "zero-shot-classification",
            model=model,
            device=device,
            use_auth_token=True,
            use_fast=False,
        )

        self.initialized = True
