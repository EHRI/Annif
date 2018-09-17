"""Ensemble backend that combines results from multiple projects"""


import annif.hit
import annif.project
import annif.util
from . import backend


class EnsembleBackend(backend.AnnifBackend):
    """Ensemble backend that combines results from multiple projects"""
    name = "ensemble"

    def _analyze_with_sources(self, text, sources):
        hits_from_sources = []
        for project_id, weight in sources:
            project = annif.project.get_project(project_id)
            hits = [hit for hit in project.analyze(text) if hit.score > 0.0]
            self.debug(
                'Got {} hits from project {}'.format(
                    len(hits), project.project_id))
            hits_from_sources.append(
                annif.hit.WeightedHits(
                    hits=hits, weight=weight))
        return hits_from_sources

    def _analyze(self, text, project, params):
        sources = annif.util.parse_sources(params['sources'])
        hits_from_sources = self._analyze_with_sources(text, sources)
        merged_hits = annif.util.merge_hits(hits_from_sources)
        self.debug('{} hits after merging'.format(len(merged_hits)))
        return merged_hits