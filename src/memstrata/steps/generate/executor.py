"""MediaGenerator executor: backend.generate(task) → GenerationArtifact."""

from __future__ import annotations

from typing import Protocol

from memstrata.steps.generate.schemas import GenerationArtifact, MediaGenerationTask


class MediaGenerationBackend(Protocol):
    def generate(self, task: MediaGenerationTask) -> GenerationArtifact:
        """Generate one archived artifact from an approved task."""


class MediaGenerator:
    """Thin executor: composition stays outside; backends only see MediaGenerationTask."""

    def __init__(self, backend: MediaGenerationBackend) -> None:
        self.backend = backend

    def execute(self, task: MediaGenerationTask) -> GenerationArtifact:
        return self.backend.generate(task)
