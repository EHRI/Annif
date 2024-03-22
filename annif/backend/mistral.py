import os
import sys
import json

from . import backend
from annif.suggestion import SubjectSuggestion

import requests


class MistralBackend(backend.AnnifBackend):
    """Backend using the Mistral classifier API for subject indexing."""

    name = "mistral"
    initialized = False
    is_trained = True
    subject_id = 0
    modification_time = None
    _session = None
    _terms = None
    _terms_rev = None

    def initialize(self, parallel=False):
        super().initialize(parallel)

        self._terms = {s.uri: s.labels["en"] for i, s in self.project.vocab.subjects.active}
        self._terms_rev = {v: k for k, v in self._terms.items()}

        mistral_key = os.environ.get('MISTRAL_KEY')
        if not mistral_key:
            self.warning("MISTRAL_KEY environment variable not set")

        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {mistral_key}'
        })

        self.initialized = True

    def _suggest(self, text, params):

        limit = int(params.get("limit", 5))
        model = params.get("model", "mistral-small-latest")

        explain = f"""I'm going to provide a list of "subject headings" 
                followed by a piece of text. The text will be denoted 
                by "-----" above and below. 

                I want you to select the most appropriate {limit} subject
                headings that describe the text and return them as a 
                single JSON list of strings. Do not output anything
                other than the single JSON list, i.e. no notes or apologies.

                Here are the subject headings:"""

        labels_text = "\n".join(self._terms.values())

        prompt = f"""
                {explain}

                {labels_text}

                -----
                {text}
                -----
                """

        data = dict(
            model=model,
            messages=[
                dict(
                    role="user",
                    content=prompt
                )
            ],
            random_seed=2010 # makes results deterministic
        )

        try:
            response = self._session.post('https://api.mistral.ai/v1/chat/completions', json=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            self.warning(f"HTTP request failed: {err}")
            return []

        out = response.json()
        if out['choices']:
            try:
                text_labels = json.loads(out['choices'][0]['message']['content'])
                self.debug("Output: {}".format(text_labels))
            except json.JSONDecodeError:
                self.warning(f"Unable to parse response: {out}")
                return []

            suggestions = [
                {
                    "uri": self._terms_rev[text],
                    "label": text,
                    "notation": None,
                    "score": 0.9 - (i * 0.05) # dummy value
                } for i, text in enumerate(text_labels) if text in self._terms_rev
            ]

            self.debug("Suggestions: {}".format(suggestions))

            return [
                SubjectSuggestion(
                    subject_id=self.project.subjects.by_uri(suggestion["uri"]),
                    score=suggestion["score"]
                ) for suggestion in suggestions
            ]

        # No data, just return an empty list...
        self.warning("No suggestions")
        return []