"""Entities of the My Resume lab: the job postings that get searched."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Job(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    company: str
    location: str
    work_mode: str  # Remote / Hybrid / On-site
    level: str
    summary: str
    skills: list[str]
    experience: str  # e.g. "5+ years"
    responsibilities: list[str]
    qualifications: list[str]  # required
    nice_to_have: list[str]

    @property
    def text(self) -> str:
        """The document every retrieval method indexes: the same text for BM25 and for the embedding model."""
        return (
            f"{self.title}. {self.summary} Responsibilities: {'; '.join(self.responsibilities)}. "
            f"Required qualifications: {'; '.join(self.qualifications)}. Required skills: {', '.join(self.skills)}."
        )
