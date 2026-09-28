"""Vectorisation des textes.

`OllamaEmbedder` passe par l'API compatible OpenAI d'Ollama — le même
client `AsyncOpenAI` que le reste du projet. Aucun texte ne quitte
l'infrastructure.
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Embedder(ABC):
    dimension: int

    @abstractmethod
    async def documents(self, textes: list[str]) -> list[list[float]]:
        """Vecteurs des extraits à indexer."""

    @abstractmethod
    async def requete(self, texte: str) -> list[float]:
        """Vecteur d'une question utilisateur."""


class OllamaEmbedder(Embedder):
    """Embeddings servis par Ollama.

    `nomic-embed-text` est entraîné avec des préfixes de tâche : il
    attend « search_document: » devant un extrait et « search_query: »
    devant une question. Les omettre dégrade sensiblement la pertinence
    de la recherche — c'est la cause la plus fréquente d'un RAG qui
    « trouve mal » alors que tout semble correct.
    """

    PREFIXES = {
        "nomic-embed-text": ("search_document: ", "search_query: "),
    }

    def __init__(self, client: Any, modele: str = "nomic-embed-text",
                 dimension: int = 768, taille_lot: int = 32):
        self.client = client
        self.modele = modele
        self.dimension = dimension
        self.taille_lot = taille_lot
        base = modele.split(":")[0]
        self._prefixe_doc, self._prefixe_req = self.PREFIXES.get(base, ("", ""))

    async def _vectoriser(self, textes: list[str]) -> list[list[float]]:
        vecteurs: list[list[float]] = []
        # Par lots : une requête par extrait serait lente, une requête
        # pour tout le corpus dépasserait les limites du serveur.
        for i in range(0, len(textes), self.taille_lot):
            lot = textes[i:i + self.taille_lot]
            reponse = await self.client.embeddings.create(model=self.modele, input=lot)
            vecteurs.extend(d.embedding for d in reponse.data)

        if vecteurs and len(vecteurs[0]) != self.dimension:
            raise ValueError(
                f"Le modèle '{self.modele}' produit des vecteurs de dimension "
                f"{len(vecteurs[0])}, la base en attend {self.dimension}. "
                "Aligner RAG_DIMENSION sur le modèle, puis recréer la collection."
            )
        return vecteurs

    async def documents(self, textes: list[str]) -> list[list[float]]:
        return await self._vectoriser([f"{self._prefixe_doc}{t}" for t in textes])

    async def requete(self, texte: str) -> list[float]:
        return (await self._vectoriser([f"{self._prefixe_req}{texte}"]))[0]


class HachageEmbedder(Embedder):
    """Embeddings déterministes par hachage de mots — tests uniquement.

    Deux textes partageant des mots obtiennent des vecteurs proches :
    suffisant pour vérifier la plomberie de bout en bout sans serveur
    d'inférence. Ne mesure aucune proximité sémantique réelle.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def _vecteur(self, texte: str) -> list[float]:
        v = [0.0] * self.dimension
        for mot in texte.lower().split():
            mot = "".join(c for c in mot if c.isalnum())
            if len(mot) < 3:
                continue
            h = int(hashlib.md5(mot.encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        norme = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norme for x in v]

    async def documents(self, textes: list[str]) -> list[list[float]]:
        return [self._vecteur(t) for t in textes]

    async def requete(self, texte: str) -> list[float]:
        return self._vecteur(texte)
