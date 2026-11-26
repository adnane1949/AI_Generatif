"""Sorties structurées (Pydantic) : sources, réponse du modèle, réponse de l'API."""
from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    reference: str                      # S1, S2, ...
    id: str                             # identifiant ChromaDB
    type: Literal["texte", "image"]
    pdf: str
    page: int
    famille: str = "inconnue"
    version: str = "n/a"
    score: float = Field(description="similarité CLIP ou score du reranker")
    extrait: str | None = None          # texte du passage
    chemin_image: str | None = None     # fichier image extrait


class Reponse(BaseModel):
    """Schéma imposé au LLM (format JSON Ollama)."""
    reponse: str = Field(description="Réponse en français, fondée uniquement sur les sources")
    citations: list[str] = Field(default_factory=list, description="Références utilisées, ex. ['S1','S3']")
    incertitudes: list[str] = Field(default_factory=list, description="Points non couverts ou douteux")
    besoin_clarification: bool = Field(default=False, description="Vrai si la demande est ambiguë")
    question_clarification: str | None = Field(default=None, description="Question à poser à l'utilisateur")
    niveau_confiance: Literal["faible", "moyen", "eleve"] = "moyen"


class ReponseRAG(BaseModel):
    question: str
    reponse: Reponse
    sources: list[Source]
    avertissements: list[str] = Field(default_factory=list)
    latence_ms: int = 0


class RequeteQuestion(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=12)
    type_resultat: Literal["tous", "texte", "image"] = "tous"
