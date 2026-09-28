"""Recherche au moment de la requête — le bloc (B-I) du schéma.

    question  →  vecteur  →  recherche sémantique  →  top-k  →  contexte injecté
       ①                        ②                     ③            ④

La génération (⑤ ⑥) est assurée par les agents du package `Agent`, qui
reçoivent la recherche comme un outil.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import ResultatRecherche
from .embeddings import Embedder
from .vectorstores import BaseVectorielle

logger = logging.getLogger(__name__)


class MoteurRecherche:
    """Recherche sémantique filtrée par équipe.

    Args:
        score_minimal: en dessous, un extrait est écarté même s'il figure
            dans le top-k. Mieux vaut répondre « je ne trouve pas » que
            fonder une réponse sur un extrait sans rapport.
    """

    def __init__(self, embedder: Embedder, base: BaseVectorielle, score_minimal: float = 0.0):
        self.embedder = embedder
        self.base = base
        self.score_minimal = score_minimal

    async def recherche_semantique(
        self, requete: str, top_k: int = 5, equipes: list[str] | None = None
    ) -> list[ResultatRecherche]:
        if not requete or not requete.strip():
            return []
        vecteur = await self.embedder.requete(requete.strip())
        resultats = await self.base.rechercher(vecteur, top_k=top_k, equipes=equipes)
        retenus = [r for r in resultats if r.score >= self.score_minimal]
        logger.info("Recherche « %s » : %d extrait(s), équipes %s",
                    requete[:60], len(retenus), equipes)
        return retenus

    # -----------------------------------------------------------------
    # Branchement sur le package Agent
    # -----------------------------------------------------------------
    def outils_pour_equipes(self, equipes: list[str]) -> Any:
        """Registre d'outils prêt pour `LoopAgentic`, `PydanticAgent`…

        Les équipes sont figées dans l'outil : le modèle peut choisir la
        requête et le nombre d'extraits, jamais le périmètre d'accès. Un
        argument « equipes » exposé au modèle serait une faille — une
        injection de prompt suffirait à l'élargir.
        """
        from Agent import Source, outils_rag

        perimetre = list(equipes)

        async def rechercher(requete: str, top_k: int = 5) -> list[Source]:
            resultats = await self.recherche_semantique(requete, top_k, perimetre)
            return [
                Source(reference=f"{r.source} · {r.chunk_id}", extrait=r.texte, score=r.score)
                for r in resultats
            ]

        return outils_rag(rechercher)

    async def outils_pour_session(self, session: Any) -> Any:
        """Registre d'outils restreint aux équipes du titulaire de la session.

        Fonctionne pour une session utilisateur comme pour un compte
        technique : c'est la même classe `llm`, les équipes viennent de
        la base dans les deux cas. Un compte sans équipe obtient un outil
        qui ne trouve rien — c'est le comportement voulu.
        """
        compte = session.user
        equipes = await compte.load_teams() if hasattr(compte, "load_teams") else compte.teams
        return self.outils_pour_equipes(list(equipes))
