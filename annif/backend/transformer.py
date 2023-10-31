
from typing import Any, Union, TYPE_CHECKING, Dict
from . import backend

from annif.project import AnnifProject
from annif.suggestion import SubjectSuggestion, SuggestionBatch

class BaseTransformerBackend(backend.AnnifBackend):
    """Base class for backends that use a transformer model. This class
    provides the common functionality for backends that use the
    transformers library """
    _zs = None
    _terms = None
    _terms_rev = None

    def initialize(self, parallel: bool = False) -> None:
        """Initialize the backend. This method should be overridden in
        subclasses to perform any initialization needed."""
        self._terms = {s.uri: s.labels["en"] for i, s in self.project.vocab.subjects.active}
        self._terms_rev = {v: k for k, v in self._terms.items()}


    def run_pipeline(
            self,
            texts: list[str],
            params: dict[str, Any] | None = None,
            num: int = 10,
    ) -> list[list[Union[tuple[str, str, float]], None]]:  # (concept_id, concept_text, confidence)
        """Recognise a term from the vocabulary in the given text
        :param text: Text to recognise
        :return: (concept_id, concept_text, confidence) or None if no match
        """
        if len(texts) == 0:
            return []

        threshold = float(params.get("threshold", 0.0))
        results = self._zs(texts, candidate_labels=list(self._terms.values()), multi_label=num > 1)

        def suggestions(result):
            num_results = len(result["labels"])
            return [(self._terms_rev[result["labels"][i]], result["labels"][i], result["scores"][i]) 
                    for i in range(min(num, num_results)) if result["scores"][i] > threshold]

        return [suggestions(r) for r in results]

    def _suggest_batch(self, texts: list[str], params: dict[str, Any]) -> list[SubjectSuggestion]|SuggestionBatch:

        if len(texts) == 0:
            return []

        num = int(params.get("limit"))

        # Empty texts must return an empty suggestion batch but the pipeline
        # will error, so we need to remove the empty values and then reinsert
        # an empty SuggestionBatch for empty texts
        non_empty = [text for text in texts if text.strip() != ""]
        non_empty_indices = [i for i, text in enumerate(texts) if text.strip() != ""]
        text_suggestions = self.run_pipeline(non_empty, num=num, params=params)

        def make_suggestion(uri, score):
            return SubjectSuggestion(subject_id=self.project.subjects.by_uri(uri), score=score)

        batch = []
        for i, suggestions in enumerate(text_suggestions):
            batch.append([make_suggestion(uri, score) for uri, _, score in suggestions])

        # reinsert an empty SuggestionBatch for empty texts
        for i in range(len(texts)):
            if i not in non_empty_indices:
                batch.insert(i, [])

        return SuggestionBatch.from_sequence(
            batch,
            self.project.subjects,
        )

