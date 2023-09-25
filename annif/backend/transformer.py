
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

        results = self._zs(texts, candidate_labels=list(self._terms.values()), multi_label=num > 1)
        for r in results:
            print(r)

        def suggestions(result):
            num_results = len(result["labels"])
            return [(self._terms_rev[result["labels"][i]], result["labels"][i], result["scores"][i]) for i in range(min(num, num_results))]

        return [suggestions(r) for r in results]

    def _suggest_batch(self, texts: list[str], params: dict[str, Any]) -> list[SubjectSuggestion]:

        num = int(params.get("limit"))

        text_suggestions = self.run_pipeline(texts, num=num, params=params)

        def mksuggestion(uri, text, score):
            return SubjectSuggestion(subject_id=self.project.subjects.by_uri(uri), score=score)

        batch = []
        for i, suggestions in enumerate(text_suggestions):
            batch.append([mksuggestion(uri, text, score) for uri, text, score in suggestions])

        return SuggestionBatch.from_sequence(
            batch,
            self.project.subjects,
        )

