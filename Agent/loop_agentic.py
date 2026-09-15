"""Palier 1 — boucle outillée, sans dépendance nouvelle.

Le modèle reçoit la liste des outils. À chaque tour il choisit d'en
appeler un ou de répondre. Le résultat est réinjecté et la boucle
continue jusqu'à la réponse finale ou l'épuisement du budget.

N'utilise que le client `AsyncOpenAI` déjà présent dans le projet et
Pydantic pour la validation. C'est le palier réversible : le supprimer
ne laisse aucune trace dans le reste du code.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from .base import (
    AgentError,
    BaseAgent,
    BudgetEpuise,
    SessionLLM,
    Etape,
    ReponseAgent,
    Source,
    TypeEtape,
)
from .tools import RegistreOutils

logger = logging.getLogger(__name__)

PROMPT_SYSTEME = (
    "Tu es l'assistant interne de la plateforme LLM souverain. "
    "Tu réponds en français, de façon concise et factuelle.\n"
    "Tu disposes d'outils : utilise-les dès que la question porte sur des "
    "documents ou des données internes, plutôt que de répondre de mémoire.\n"
    "Cite tes sources sous la forme [1], [2] quand tu t'appuies sur des extraits. "
    "Si les extraits ne permettent pas de répondre, dis-le plutôt que d'inventer."
)


class LoopAgentic(BaseAgent):
    """Boucle observation / décision / action.

    Args:
        client: client `AsyncOpenAI` (ou compatible) déjà configuré.
        modele: identifiant du modèle, tel qu'habilité pour l'utilisateur.
        outils: registre partagé.
        max_iterations: plafond de tours. Au-delà, `BudgetEpuise`.
        lecture_seule: n'expose que les outils sans effet de bord.
    """

    nom = "loop"

    def __init__(
        self,
        session: "SessionLLM",
        outils: RegistreOutils,
        max_iterations: int = 6,
        lecture_seule: bool = True,
        prompt_systeme: str = PROMPT_SYSTEME,
        temperature: float = 0.3,
    ):
        super().__init__(session, max_iterations)
        self.outils = outils
        self.lecture_seule = lecture_seule
        self.prompt_systeme = prompt_systeme
        # Température basse : un agent qui décide d'appeler un outil doit
        # être reproductible, pas créatif.
        self.temperature = temperature

    # -----------------------------------------------------------------
    async def run(self, question: str, historique: list[dict] | None = None) -> ReponseAgent:
        debut = self._chrono()
        # Résolu ici, et non à la construction : l'agent suit le modèle
        # que l'utilisateur a sélectionné, y compris s'il en a changé
        # entre deux messages.
        modele = await self._modele_actif()
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.prompt_systeme}]
        messages += historique or []
        messages.append({"role": "user", "content": question})

        schemas = self.outils.schemas_openai(self.lecture_seule)
        etapes: list[Etape] = []
        sources: list[Source] = []
        tokens = 0

        for tour in range(1, self.max_iterations + 1):
            reponse = await self._completion(modele, messages, schemas)
            message = reponse.choices[0].message
            usage = getattr(reponse, "usage", None)
            if usage and getattr(usage, "total_tokens", None):
                tokens += usage.total_tokens

            appels = getattr(message, "tool_calls", None)
            if not appels:
                etapes.append(Etape(numero=len(etapes) + 1, type=TypeEtape.REPONSE))
                return ReponseAgent(
                    contenu=message.content or "",
                    modele=modele,
                    etapes=etapes,
                    sources=sources,
                    iterations=tour,
                    tokens=tokens or None,
                    latence_ms=self._ms(debut),
                )

            # Le message de l'assistant portant les appels doit être
            # réinjecté tel quel : l'API exige que chaque tool_call_id
            # trouve sa réponse dans le tour suivant.
            messages.append(self._serialiser(message))

            for appel in appels:
                etape, contenu = await self._executer(appel, len(etapes) + 1)
                etapes.append(etape)
                messages.append({
                    "role": "tool",
                    "tool_call_id": appel.id,
                    "content": contenu,
                })

        raise BudgetEpuise(
            f"L'agent n'a pas conclu après {self.max_iterations} tours "
            f"(outils appelés : {', '.join(e.outil for e in etapes if e.outil) or 'aucun'}). "
            "Reformuler la question ou relever max_iterations."
        )

    # -----------------------------------------------------------------
    async def stream(self, question: str, **kwargs: Any) -> AsyncIterator[str]:
        """Restitution en deux temps.

        Les tours d'outils ne produisent pas de texte affichable : on
        annonce l'action en cours, puis on rend la réponse. Streamer la
        dernière génération token par token demanderait de dupliquer la
        boucle ; le gain ne le justifie pas au palier 1.
        """
        reponse = await self.run(question, **kwargs)
        for etape in reponse.etapes:
            if etape.type is TypeEtape.OUTIL:
                yield f"_{etape.outil} ({etape.duree_ms} ms)_\n\n"
        yield reponse.contenu

    # -----------------------------------------------------------------
    async def _completion(self, modele: str, messages: list[dict], schemas: list[dict]):
        params: dict[str, Any] = {
            "model": modele,
            "messages": messages,
            "temperature": self.temperature,
        }
        if schemas:
            params["tools"] = schemas
            params["tool_choice"] = "auto"
        try:
            return await self.client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"Appel au modèle '{modele}' impossible : {exc}") from exc

    # -----------------------------------------------------------------
    async def _executer(self, appel: Any, numero: int) -> tuple[Etape, str]:
        """Exécute un appel d'outil.

        Les erreurs sont renvoyées **au modèle** plutôt que levées : un
        argument invalide est une information exploitable, il corrige et
        rappelle l'outil. Faire échouer toute la requête pour un
        paramètre mal formé serait disproportionné.
        """
        nom = appel.function.name
        depart = self._chrono()

        try:
            arguments = json.loads(appel.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            message = f"Arguments illisibles : {exc}"
            return (
                Etape(numero=numero, type=TypeEtape.ERREUR, outil=nom, observation=message),
                message,
            )

        outil = self.outils.get(nom)
        if outil is None:
            message = (
                f"Outil '{nom}' inconnu. Disponibles : "
                + ", ".join(o.nom for o in self.outils.lister(self.lecture_seule))
            )
            return (
                Etape(numero=numero, type=TypeEtape.ERREUR, outil=nom,
                      arguments=arguments, observation=message),
                message,
            )
        if self.lecture_seule and not outil.lecture_seule:
            message = f"Outil '{nom}' indisponible : la session est en lecture seule."
            return (
                Etape(numero=numero, type=TypeEtape.ERREUR, outil=nom,
                      arguments=arguments, observation=message),
                message,
            )

        try:
            resultat = await outil.appeler(arguments)
            contenu = resultat if isinstance(resultat, str) else json.dumps(
                resultat, ensure_ascii=False, default=str
            )
            etape = Etape(numero=numero, type=TypeEtape.OUTIL, outil=nom,
                          arguments=arguments, observation=contenu,
                          duree_ms=self._ms(depart))
            logger.info("outil %s en %s ms", nom, etape.duree_ms)
            return etape, contenu

        except ValueError as exc:  # validation Pydantic
            message = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Échec de l'outil %s", nom)
            message = f"L'outil '{nom}' a échoué : {exc}"

        return (
            Etape(numero=numero, type=TypeEtape.ERREUR, outil=nom,
                  arguments=arguments, observation=message, duree_ms=self._ms(depart)),
            message,
        )

    # -----------------------------------------------------------------
    @staticmethod
    def _serialiser(message: Any) -> dict[str, Any]:
        """Normalise le message de l'assistant pour le renvoyer à l'API."""
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": a.id,
                    "type": "function",
                    "function": {"name": a.function.name, "arguments": a.function.arguments},
                }
                for a in message.tool_calls
            ],
        }
