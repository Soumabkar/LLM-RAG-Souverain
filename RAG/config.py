"""Configuration et objets d'échange du package RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TypeDocument(str, Enum):
    """Formats acceptés à l'ingestion."""

    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    PDF = "pdf"
    PARQUET = "parquet"
    AVRO = "avro"


EXTENSIONS: dict[str, TypeDocument] = {
    ".html": TypeDocument.HTML,
    ".htm": TypeDocument.HTML,
    ".md": TypeDocument.MARKDOWN,
    ".markdown": TypeDocument.MARKDOWN,
    ".csv": TypeDocument.CSV,
    ".pdf": TypeDocument.PDF,
    ".parquet": TypeDocument.PARQUET,
    ".avro": TypeDocument.AVRO,
}


def _env(nom: str, defaut: str) -> str:
    return os.getenv(nom, defaut).strip()


@dataclass(frozen=True)
class ConfigRAG:
    """Paramètres du RAG, surchargeables par variables d'environnement.

    Le nom du bucket suit les règles S3 : minuscules, chiffres et tirets.
    « RAG_Souverain » serait rejeté par MinIO — ni majuscule, ni tiret bas.
    """

    bucket: str = "rag-souverain"
    prefixe_brute: str = "data-brute"
    prefixe_processing: str = "data-processing"
    prefixe_corpus: str = "data-corpus"
    prefixe_chunking: str = "data-chunking"

    taille_chunk: int = 1000
    chevauchement: int = 150

    modele_embedding: str = "nomic-embed-text"
    dimension: int = 768

    @classmethod
    def depuis_env(cls) -> "ConfigRAG":
        return cls(
            bucket=_env("RAG_BUCKET", cls.bucket),
            taille_chunk=int(_env("RAG_TAILLE_CHUNK", str(cls.taille_chunk))),
            chevauchement=int(_env("RAG_CHEVAUCHEMENT", str(cls.chevauchement))),
            modele_embedding=_env("RAG_MODELE_EMBEDDING", cls.modele_embedding),
            dimension=int(_env("RAG_DIMENSION", str(cls.dimension))),
        )


# =====================================================================
# Objets d'échange
# =====================================================================
class DocumentBrut(BaseModel):
    """Un document déposé en zone brute.

    `equipes` porte le contrôle d'accès : il est propagé jusqu'aux
    chunks, et la recherche ne remonte que les extraits des équipes de
    l'appelant. Sans ce champ, le RAG rendrait à tout le monde tout ce
    qui a été indexé.
    """

    doc_id: str
    nom: str
    type: TypeDocument
    cle: str
    equipes: list[str] = Field(default_factory=list)
    sha256: str
    taille: int
    depose_le: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RapportSecurisation(BaseModel):
    """Ce que le traitement a masqué — pour l'audit, sans les valeurs."""

    emails: int = 0
    telephones: int = 0
    ibans: int = 0
    cartes: int = 0
    nir: int = 0
    caracteres_retires: int = 0

    @property
    def total_masque(self) -> int:
        return self.emails + self.telephones + self.ibans + self.cartes + self.nir


class Chunk(BaseModel):
    """Un extrait prêt à vectoriser."""

    chunk_id: str
    doc_id: str
    index: int = Field(ge=0)
    texte: str
    equipes: list[str] = Field(default_factory=list)
    source: str
    type: TypeDocument


class ResultatRecherche(BaseModel):
    """Un extrait remonté par la recherche sémantique."""

    chunk_id: str
    doc_id: str
    texte: str
    source: str
    score: float
    equipes: list[str] = Field(default_factory=list)


@dataclass
class BilanIndexation:
    """Résultat d'une indexation de bout en bout."""

    document: DocumentBrut
    caracteres_extraits: int = 0
    securisation: RapportSecurisation = field(default_factory=RapportSecurisation)
    chunks: int = 0
    vecteurs: int = 0
