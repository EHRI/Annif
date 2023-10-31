"""Unit tests for the fastText backend in Annif"""

import pytest

import annif.backend
import annif.corpus

mdeberta = pytest.importorskip("annif.backend.mdeberta")


def test_mdeberta_default_params(project):
    mdeberta_type = annif.backend.get_backend("mdeberta")
    mdeberta = mdeberta_type(backend_id="mdeberta", config_params={}, project=project)

    expected_default_params = {"limit": 100}
    actual_params = mdeberta.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_mdeberta_suggest(project):
    mdeberta_type = annif.backend.get_backend("mdeberta")
    mdeberta = mdeberta_type(
        backend_id="mdeberta",
        config_params={
            "limit": 5,
        },
        project=project,
    )

    results = mdeberta.suggest(
        [
            """Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan."""
        ]
    )[0]

    assert len(results) > 0
    archaeology = project.subjects.by_uri("http://www.yso.fi/onto/yso/p1265")
    assert archaeology in [result.subject_id for result in results]


def test_mdeberta_suggest_empty_chunks(project):
    mdeberta_type = annif.backend.get_backend("mdeberta")
    mdeberta = mdeberta_type(
        backend_id="mdeberta",
        config_params={
            "limit": 100,
        },
        project=project,
    )

    results = mdeberta.suggest([""])

    assert len(results) == 1
    assert results[0] == []
