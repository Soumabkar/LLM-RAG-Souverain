"""Classe `llm` : jonction entre un utilisateur authentifié et un modèle.

`ai_model` sait dialoguer avec un LLM, `user` sait qui est connecté. `llm`
associe les deux : c'est le point d'entrée de Chainlit, et l'endroit où se
brancheront plus tard les quotas, la journalisation des usages et le
filtrage documentaire par équipe.

Une instance = une conversation d'un utilisateur. L'historique vit dans
`ai_model`, pas ici : le dupliquer ferait diverger les deux.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from Model.ai_model import ChatResponse, ai_model
from Engine.models import user

logger = logging.getLogger(__name__)


class LLMAccessError(PermissionError):
    """Le compte n'existe pas ou n'est rattaché à aucune équipe."""


class llm:
    def __init__(self, model: ai_model, user: user, require_team: bool = False):
        self.model = model
        self.user = user
        # Certains déploiements exigent un rattachement pour ouvrir l'accès ;
        # d'autres acceptent un compte isolé. À toi de trancher.
        self.require_team = require_team
        self._authorized = False

    def __repr__(self) -> str:  # pragma: no cover
        return f"llm(model={self.model.model!r}, user={self.user.email!r})"

    # -----------------------------------------------------------------
    async def authorize(self) -> None:
        """Vérifie que le compte existe encore en base.

        Contrôlé à chaque conversation, pas une fois pour toutes : un
        compte supprimé pendant une session ouverte doit perdre l'accès.
        """
        if await self.user.find(self.user.email) is None:
            raise LLMAccessError(f"Le compte '{self.user.email}' n'existe pas.")

        teams = await self.user.load_teams()
        if self.require_team and not teams:
            raise LLMAccessError(
                f"Le compte '{self.user.email}' n'est rattaché à aucune équipe."
            )

        self._authorized = True

    # -----------------------------------------------------------------
    def _system_prompt(self) -> str:
        """Prompt système enrichi du contexte utilisateur."""
        equipes = ", ".join(self.user.teams) if self.user.teams else "aucune"
        return (
            "Tu es l'assistant interne de la plateforme LLM souverain. "
            "Tu réponds en français, de façon concise et factuelle. "
            "Si tu ne sais pas, tu le dis.\n"
            f"Utilisateur : {self.user.login} ({self.user.email}). "
            f"Équipes : {equipes}."
        )

    # -----------------------------------------------------------------
    async def response_llm(self, message: str) -> str:
        """Génère une réponse du modèle pour le message fourni.

        Args:
            message: la question de l'utilisateur.

        Returns:
            La réponse du modèle.

        Raises:
            LLMAccessError: le compte n'est pas (ou plus) autorisé.
            AIModelError:   le modèle est injoignable ou la requête invalide.
        """
        if not self._authorized:
            await self.authorize()
            self.model.system_prompt = self._system_prompt()

        reponse: ChatResponse = await self.model.initialyse_chat(message)
        self._trace(reponse)
        return reponse.content

    # -----------------------------------------------------------------
    async def stream_response_llm(self, message: str) -> AsyncIterator[str]:
        """Même chose, token par token, pour l'affichage Chainlit."""
        if not self._authorized:
            await self.authorize()
            self.model.system_prompt = self._system_prompt()

        async for token in self.model.stream_chat(message):
            yield token

    # -----------------------------------------------------------------
    def _trace(self, reponse: ChatResponse) -> None:
        """Trace l'usage. Point d'accroche pour une table llm_usage."""
        logger.info(
            "llm | %s | équipes=%s | modèle=%s | %s tokens | %s ms",
            self.user.email,
            ",".join(self.user.teams) or "-",
            reponse.model,
            reponse.completion_tokens,
            reponse.latency_ms,
        )

    # -----------------------------------------------------------------
    def reset(self) -> None:
        """Vide l'historique sans perdre l'autorisation ni le contexte."""
        self.model.reset(self._system_prompt())

    def response_llm_rag(self):
        """
        Méthode pour générer une réponse à partir du modèle LLM avec RAG.
        Returns:
            str: La réponse générée par le modèle LLM avec RAG.
        """
        pass
        
        
    