from typing import Union, Any

from . import backend
from annif.corpus.types import DocumentCorpus
from annif.suggestion import SubjectSuggestion


class XlmRobertaBackend(backend.AnnifBackend):
    """Recogniser using the joeddav/xlm-roberta-large-xnli from HuggingFace."""
    name = "xlmroberta"
    initialized = False
    subject_id = 0
    is_trained = True
    modification_time = None
    _zs = None

    def initialize(self, parallel: bool = False) -> None:
        from transformers import pipeline
        model = "joeddav/xlm-roberta-large-xnli"

        from torch import cuda
        device = 0 if cuda.is_available() else -1
        self._zs = pipeline("zero-shot-classification", model=model, device=device, use_auth_token=True)

        self.initialized = True

    def run_pipeline(self, text: str, num: int = 10) -> list[Union[tuple[str, str, float], None]]:  # (concept_id, concept_text, confidence)
        """Recognise a term from the vocabulary in the given text
        :param text: Text to recognise
        :param zs: Zero-shot pipeline
        :param vocab: the Vocabulary dictionary
        :return: (concept_id, concept_text, confidence) or None if no match
        """
        terms = {s.uri: s.labels["en"] for i, s in self.project.vocab.subjects.active}
        terms_rev = {v: k for k, v in terms.items()}
        result = self._zs(text, candidate_labels=list(terms.values()), multi_label=num > 1)
        num_results = len(result["labels"])
        return [(terms_rev[result["labels"][i]], result["labels"][i], result["scores"][i]) for i in range(min(num, num_results))]

    def _suggest(self, text: str, params: dict[str, Any]) -> list[SubjectSuggestion]:

        # Ensure tests fail if "text" with wrong type ends up here
        assert isinstance(text, str)

        # Give no hits for no text
        if len(text) == 0:
            return []

        print(f"Suggest! {text}")

        suggestions = self.run_pipeline(text)
        return [SubjectSuggestion(subject_id=self.project.subjects.by_uri(uri), score=score) for uri, _, score in suggestions]

    def _learn(
            self,
            corpus: DocumentCorpus,
            params: dict[str, Any],
    ) -> None:
        # in this dummy backend we "learn" by picking up the subject ID
        # of the first subject of the first document in the learning set
        # and using that in subsequent analysis results
        for doc in corpus.documents:
            if doc.subject_set:
                self.subject_id = doc.subject_set[0]
            break

