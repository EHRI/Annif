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
    session = None
    _terms = None
    _terms_rev = None

    def initialize(self, parallel=False):
        super().initialize(parallel)

        self._terms = {s.uri: s.labels["en"] for i, s in self.project.vocab.subjects.active}
        self._terms_rev = {v: k for k, v in self._terms.items()}

        mistral_key = os.environ.get('MISTRAL_KEY')

        self.session = requests.Session()
        self.session.headers.update({
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
                headings that apply to the text and output them 
                as a JSON list of strings. Do not output anything
                other than the JSON list, i.e. no notes or apologies.

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
            response = self.session.post('https://api.mistral.ai/v1/chat/completions', json=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            print(f"HTTP request failed: {err}", file=sys.stderr)
            return []

        out = response.json()
        if out['choices']:
            try:
                text_labels = json.loads(out['choices'][0]['message']['content'])
                print("Output: {}".format(text_labels), file=sys.stderr)
            except json.JSONDecodeError:
                print(f"Unable to parse response: {out}", file=sys.stderr)
                return []

            suggestions = [
                {
                    "uri": self._terms_rev[text],
                    "label": text,
                    "notation": None,
                    "score": 0.9 - (i * 0.05) # dummy value
                } for i, text in enumerate(text_labels) if text in self._terms_rev
            ]

            print("Suggestions: {}".format(suggestions), file=sys.stderr)

            return [
                SubjectSuggestion(
                    subject_id=self.project.subjects.by_uri(suggestion["uri"]),
                    score=suggestion["score"]
                ) for suggestion in suggestions
            ]

        # No data, just return an empty list...
        print("No suggestions", file=sys.stderr)
        return []