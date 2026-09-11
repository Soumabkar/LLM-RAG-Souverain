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

from Model.ai_model import (
    AIModelError,
    ChatResponse,
    ai_model,
    installed_models,
    matches_installed,
)
from Engine.models import user

logger = logging.getLogger(__name__)


class LLMAccessError(PermissionError):
    """Le compte n'existe pas, n'a pas d'équipe, ou n'a pas droit au modèle."""


class LLMModelUnavailable(RuntimeError):
    """Le modèle est autorisé, mais absent du serveur d'inférence.

    Distinct de `LLMAccessError` : ce n'est pas un refus de droits mais un
    écart entre le catalogue et le parc installé. Le message doit orienter
    vers l'administrateur, pas vers une demande d'habilitation.
    """


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

        # Le modèle courant doit être dans les habilitations de l'utilisateur.
        if not await self.user.can_use_model(self.model.model):
            raise LLMAccessError(
                f"Le modèle '{self.model.model}' n'est pas autorisé pour "
                f"'{self.user.email}'."
            )

        self._authorized = True

    # -----------------------------------------------------------------
    def _base_url(self) -> str | None:
        base = getattr(self.model.client, "base_url", None)
        return str(base) if base else None

    async def installed(self) -> set[str] | None:
        """Modèles présents sur le serveur, ou None s'il est injoignable."""
        try:
            return await installed_models(self._base_url())
        except AIModelError as exc:
            logger.warning("Inventaire du serveur d'inférence indisponible : %s", exc)
            return None

    async def available_models(self, only_installed: bool = True) -> list[dict]:
        """Modèles que cet utilisateur peut choisir.

        Par défaut, l'intersection entre ses habilitations et ce qui est
        réellement chargé sur le serveur : proposer un modèle cataloguté
        mais désinstallé revient à offrir un bouton qui renvoie une 404.

        Si le serveur est injoignable, la liste complète est renvoyée —
        une liste vide priverait l'utilisateur de tout choix alors que le
        problème est ailleurs et sans doute temporaire.
        """
        modeles = await self.user.allowed_models()
        if not only_installed:
            return modeles

        disponibles = await self.installed()
        if disponibles is None:
            return modeles

        return [m for m in modeles if matches_installed(m["code_model"], disponibles)]

    async def switch_model(self, code_model: str) -> None:
        """Change de modèle sans perdre la conversation.

        Deux contrôles, dans cet ordre : l'habilitation, puis la présence
        sur le serveur. Ils sont refaits ici et pas seulement à
        l'affichage — la liste envoyée à l'IHM est un confort, rien
        n'empêche un client de demander autre chose.
        """
        if code_model == self.model.model:
            return

        if not await self.user.can_use_model(code_model):
            raise LLMAccessError(
                f"Le modèle '{code_model}' n'est pas autorisé pour '{self.user.email}'."
            )

        disponibles = await self.installed()
        if disponibles is not None and not matches_installed(code_model, disponibles):
            raise LLMModelUnavailable(
                f"Le modèle '{code_model}' est autorisé mais n'est pas installé "
                "sur le serveur d'inférence. Signale-le à un administrateur."
            )

        ancien = self.model.model
        self.model.model = code_model
        logger.info("%s : modèle %s -> %s", self.user.email, ancien, code_model)

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