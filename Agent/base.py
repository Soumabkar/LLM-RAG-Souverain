"""Socle commun aux trois implémentations d'agent.

Les trois classes — `LoopAgentic`, `PydanticAgent`, `LangGraphAgent` —
exposent la même interface et manipulent les mêmes objets. Passer de
l'une à l'autre est un changement de configuration, pas une réécriture :
c'est ce qui rend les trois paliers de la feuille de route
interchangeables plutôt que successifs.

Tous les objets d'échange sont des modèles Pydantic : une réponse mal
formée échoue à la frontière du module, pas trois couches plus loin.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator


class AgentError(RuntimeError):
    """Échec non récupérable de l'agent."""


class AccesRefuse(AgentError):
    """Le modèle courant n'est pas autorisé pour cet utilisateur.

    Distinct d'une panne : l'agent fonctionne, c'est l'habilitation qui
    manque. Le message doit orienter vers l'administrateur, pas vers le
    support technique.
    """


class BudgetEpuise(AgentError):
    """Nombre maximum d'itérations atteint sans réponse finale.

    Distinct d'une erreur technique : l'agent fonctionne, mais il tourne
    en rond. Le message doit le dire, sinon on cherche une panne qui
    n'existe pas.
    """


# =====================================================================
# Source du modèle courant
# =====================================================================
@runtime_checkable
class SessionLLM(Protocol):
    """Ce que l'agent attend d'une session — le contrat de la classe `llm`.

    L'agent reçoit une session, pas un client et un identifiant de
    modèle : c'est elle qui porte l'identité, les droits et le moteur
    d'inférence. Un agent sous compte de service et un agent sous compte
    utilisateur ne diffèrent alors que par la session qu'on leur passe.

    Déclaré en protocole plutôt qu'en import direct de `llm` : le package
    Agent reste testable sans PostgreSQL, et la couche `LLM` peut importer
    `Agent` sans créer de cycle.
    """

    @property
    def client(self) -> Any:
        """Client d'inférence, compatible OpenAI."""
        ...

    async def modele_courant(self) -> str:
        """Identifiant du modèle actuellement sélectionné."""
        ...

    async def autorise(self, code_model: str) -> bool:
        """Vrai si le titulaire de la session a le droit d'employer ce modèle."""
        ...


# Ancien nom, conservé pour ne pas casser le code existant.
FournisseurModele = SessionLLM


# =====================================================================
# Objets d'échange
# =====================================================================
class TypeEtape(str, Enum):
    REFLEXION = "reflexion"
    OUTIL = "outil"
    REPONSE = "reponse"
    ERREUR = "erreur"


class Etape(BaseModel):
    """Une étape du raisonnement, tracée pour l'audit et le débogage."""

    numero: int = Field(ge=1)
    type: TypeEtape
    outil: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    duree_ms: int = Field(default=0, ge=0)
    horodatage: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("observation")
    @classmethod
    def _tronquer(cls, v: str | None) -> str | None:
        """Une observation d'outil peut faire des milliers de lignes.

        On tronque pour la trace, pas pour le modèle : ce champ sert au
        journal et à l'affichage, la valeur complète part dans le
        contexte du LLM.
        """
        if v and len(v) > 2000:
            return v[:2000] + f"… [{len(v) - 2000} caractères tronqués]"
        return v

    def __str__(self) -> str:  # pragma: no cover
        if self.type is TypeEtape.OUTIL:
            return f"{self.numero}. {self.outil}({self.arguments}) → {self.duree_ms} ms"
        return f"{self.numero}. {self.type.value}"


class Source(BaseModel):
    """Un extrait documentaire ayant servi à la réponse."""

    reference: str
    extrait: str = ""
    score: float | None = None


class ReponseAgent(BaseModel):
    """Résultat d'une exécution."""

    contenu: str
    modele: str
    etapes: list[Etape] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    iterations: int = Field(default=0, ge=0)
    tokens: int | None = None
    latence_ms: int = Field(default=0, ge=0)

    @property
    def outils_appeles(self) -> list[str]:
        return [e.outil for e in self.etapes if e.type is TypeEtape.OUTIL and e.outil]

    def __str__(self) -> str:  # pragma: no cover
        return self.contenu


# =====================================================================
# Interface
# =====================================================================
class BaseAgent(ABC):
    """Contrat commun aux trois implémentations.

    `nom` sert à la journalisation et permet à l'interface d'afficher
    quelle implémentation a produit une réponse — utile pendant la phase
    où les trois coexistent.
    """

    nom: str = "agent"

    def __init__(self, session: SessionLLM, max_iterations: int = 6):
        """
        Args:
            session: session `llm` — utilisateur interactif ou compte de
                service. L'agent en tire son client, son modèle et ses
                droits.
            max_iterations: plafond de tours.
        """
        self.session = session
        # Le modèle n'est pas figé à la construction : il est résolu à
        # chaque exécution auprès de la session. Sans cela, un changement
        # de modèle dans l'interface ne s'appliquerait qu'à la prochaine
        # ouverture de session — l'agent continuerait sur l'ancien.
        self._modele_fige: str | None = None

        # Un plafond est indispensable : un agent qui boucle consomme des
        # jetons sans produire de réponse. Six tours couvrent largement un
        # enchaînement recherche → lecture → synthèse.
        self.max_iterations = max_iterations

    # -----------------------------------------------------------------
    @property
    def client(self) -> Any:
        """Client d'inférence de la session."""
        return self.session.client

    @property
    def modele(self) -> str:
        """Dernier modèle résolu — pour la journalisation et la trace.

        Avant la première exécution, la valeur n'est pas encore connue :
        on renvoie une chaîne parlante plutôt que None, qui finirait dans
        un message d'erreur.
        """
        return self._modele_fige or "(non résolu)"

    @property
    def titulaire(self) -> str:
        """Compte sous lequel l'agent s'exécute — humain ou technique."""
        compte = getattr(self.session, "user", None)
        return getattr(compte, "email", "(inconnu)")

    async def _modele_actif(self) -> str:
        """Résout le modèle courant et vérifie l'habilitation.

        Le contrôle est refait à chaque exécution, pas mis en cache : une
        habilitation retirée en cours de session doit s'appliquer au
        message suivant, pas à la reconnexion.
        """
        code = await self.session.modele_courant()
        if not await self.session.autorise(code):
            raise AccesRefuse(
                f"Le modèle « {code} » n'est pas autorisé pour « {self.titulaire} » : "
                "l'agent ne peut pas être exécuté. Contactez un administrateur "
                "pour l'habilitation de l'équipe concernée."
            )
        self._modele_fige = code  # mémorisé pour la trace uniquement
        return code

    @abstractmethod
    async def run(self, question: str, **kwargs: Any) -> ReponseAgent:
        """Exécute l'agent et renvoie la réponse complète."""

    async def stream(self, question: str, **kwargs: Any) -> AsyncIterator[str]:
        """Restitution progressive.

        Implémentation par défaut : exécute puis rend le résultat d'un
        bloc. Les agents qui savent streamer redéfinissent la méthode.
        """
        reponse = await self.run(question, **kwargs)
        yield reponse.contenu

    # -----------------------------------------------------------------
    @staticmethod
    def _chrono() -> float:
        return time.perf_counter()

    @staticmethod
    def _ms(depart: float) -> int:
        return int((time.perf_counter() - depart) * 1000)
