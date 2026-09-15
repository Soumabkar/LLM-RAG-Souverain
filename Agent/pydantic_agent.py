"""Palier 2 — agent typé et observable (Pydantic AI).

Apport par rapport au palier 1 :

* **Sortie structurée garantie.** L'agent déclare un `output_type` ;
  la réponse du modèle est validée avant d'être rendue. Fini le parsing
  de texte libre pour récupérer une catégorie ou un niveau de confiance.
* **Outils typés par signature.** La signature Python devient le schéma
  transmis au modèle : plus de description à maintenir en double.
* **Observabilité native.** Instrumentation OpenTelemetry conforme aux
  conventions sémantiques, donc exploitable dans un collecteur existant.
* **Indépendance du fournisseur.** Le client `AsyncOpenAI` du projet est
  injecté tel quel : Ollama reste le moteur, rien ne part vers un tiers.

Dépendance : ``pip install "pydantic-ai-slim[openai]"``
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from .base import (
    AgentError,
    BaseAgent,
    Etape,
    SessionLLM,
    ReponseAgent,
    Source,
    TypeEtape,
)
from .tools import RegistreOutils

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Tu es l'assistant interne de la plateforme LLM souverain. "
    "Tu réponds en français, de façon concise et factuelle.\n"
    "Utilise les outils dès que la question porte sur des documents ou des "
    "données internes. Renseigne honnêtement le niveau de confiance : une "
    "réponse incertaine annoncée comme telle vaut mieux qu'une affirmation "
    "fausse."
)


class ReponseStructuree(BaseModel):
    """Format de sortie imposé au modèle.

    C'est l'intérêt principal du palier : l'interface peut afficher un
    bandeau d'avertissement sous un certain seuil de confiance, ou
    refuser de répondre sans source — décisions impossibles à prendre
    de façon fiable sur du texte libre.
    """

    reponse: str = Field(description="La réponse à la question posée")
    sources: list[str] = Field(
        default_factory=list,
        description="Références des extraits utilisés, vide si aucun",
    )
    confiance: float = Field(
        ge=0.0, le=1.0,
        description="0 si la réponse repose sur des suppositions, 1 si elle est étayée",
    )


class PydanticAgent(BaseAgent):
    """Agent Pydantic AI branché sur le client du projet.

    Args:
        client: client `AsyncOpenAI` déjà configuré (Ollama).
        modele: identifiant du modèle habilité pour l'utilisateur.
        outils: registre partagé avec les autres agents.
        structure: impose `ReponseStructuree` en sortie. Passer False
            rend une chaîne simple, pour un usage conversationnel.
    """

    nom = "pydantic-ai"

    def __init__(
        self,
        session: "SessionLLM",
        outils: RegistreOutils,
        max_iterations: int = 6,
        lecture_seule: bool = True,
        structure: bool = True,
        instructions: str = INSTRUCTIONS,
        modele_pydantic_ai: Any = None,
    ):
        super().__init__(session, max_iterations)
        self.outils = outils
        self.lecture_seule = lecture_seule
        self.structure = structure

        from pydantic_ai import Agent

        # `modele_pydantic_ai` permet d'injecter un TestModel ou un
        # FunctionModel : la suite de tests n'a pas besoin d'un serveur.
        self._modele_injecte = modele_pydantic_ai
        # Un modèle Pydantic AI par identifiant, construit à la demande :
        # l'agent n'est créé qu'une fois, mais peut servir plusieurs
        # modèles au fil des bascules de l'utilisateur.
        self._cache_modeles: dict[str, Any] = {}

        self._agent = Agent(
            None,
            instructions=instructions,
            output_type=ReponseStructuree if structure else str,
            retries=2,
        )

        for outil in self.outils.lister(self.lecture_seule):
            # tool_plain : l'outil n'a pas besoin du contexte d'exécution.
            # Le nom vient du registre, pas du nom de la fonction Python,
            # pour rester cohérent avec les deux autres implémentations.
            self._agent.tool_plain(
                outil.fonction, name=outil.nom, description=outil.description
            )

    # -----------------------------------------------------------------
    @staticmethod
    def _modele_openai(client: Any, modele: str):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(modele, provider=OpenAIProvider(openai_client=client))

    # -----------------------------------------------------------------
    def _modele_effectif(self, code: str) -> Any:
        if self._modele_injecte is not None:
            return self._modele_injecte
        if code not in self._cache_modeles:
            self._cache_modeles[code] = self._modele_openai(self.client, code)
        return self._cache_modeles[code]

    async def run(self, question: str, historique: list | None = None) -> ReponseAgent:
        debut = self._chrono()
        code = await self._modele_actif()
        try:
            # Le modèle est passé à l'exécution, pas à la construction :
            # c'est ce qui permet à l'agent de suivre les bascules.
            resultat = await self._agent.run(
                question, message_history=historique, model=self._modele_effectif(code)
            )
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"Agent Pydantic AI en échec : {exc}") from exc

        sortie = resultat.output
        if isinstance(sortie, ReponseStructuree):
            contenu = sortie.reponse
            sources = [Source(reference=r) for r in sortie.sources]
            confiance = sortie.confiance
        else:
            contenu, sources, confiance = str(sortie), [], None

        etapes = self._tracer(resultat)
        # `usage` est un attribut dans les versions récentes, une méthode
        # dans les anciennes : on accepte les deux.
        usage = resultat.usage
        if callable(usage):
            usage = usage()

        reponse = ReponseAgent(
            contenu=contenu,
            modele=code,
            etapes=etapes,
            sources=sources,
            iterations=sum(1 for e in etapes if e.type is TypeEtape.OUTIL) + 1,
            tokens=getattr(usage, "total_tokens", None),
            latence_ms=self._ms(debut),
        )
        if confiance is not None:
            logger.info("confiance déclarée : %.2f", confiance)
        return reponse

    # -----------------------------------------------------------------
    @staticmethod
    def _tracer(resultat: Any) -> list[Etape]:
        """Reconstruit la trace à partir de l'historique de messages.

        Pydantic AI conserve chaque appel d'outil et son retour : on les
        remappe sur `Etape` pour que les trois agents produisent la même
        structure de trace.
        """
        etapes: list[Etape] = []
        for message in resultat.all_messages():
            for part in getattr(message, "parts", []):
                type_part = part.__class__.__name__
                if type_part == "ToolCallPart":
                    # `final_result` est l'outil interne par lequel
                    # Pydantic AI récupère la sortie structurée : le faire
                    # apparaître dans la trace laisserait croire que
                    # l'agent a appelé un outil métier de plus.
                    if part.tool_name == "final_result":
                        continue
                    arguments = part.args
                    if isinstance(arguments, str):
                        arguments = {"_brut": arguments}
                    etapes.append(Etape(
                        numero=len(etapes) + 1, type=TypeEtape.OUTIL,
                        outil=part.tool_name, arguments=arguments or {},
                    ))
                elif type_part == "ToolReturnPart" and etapes:
                    etapes[-1].observation = str(part.content)
        etapes.append(Etape(numero=len(etapes) + 1, type=TypeEtape.REPONSE))
        return etapes

    # -----------------------------------------------------------------
    @property
    def agent(self) -> Any:
        """Agent sous-jacent, pour les usages avancés (instrumentation)."""
        return self._agent
