"""Outils mis à disposition des agents.

Un outil est déclaré **une seule fois** ici, avec son schéma d'arguments
Pydantic, et les trois implémentations le consomment : `LoopAgentic` le
traduit en schéma JSON pour l'API OpenAI, `PydanticAgent` l'enregistre
par sa signature, `LangGraphAgent` l'appelle depuis un nœud.

Sans ce registre, chaque agent redéfinirait ses outils et les trois
divergeraient au premier changement.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field, ValidationError

from .base import Source

logger = logging.getLogger(__name__)


# =====================================================================
# Schémas d'arguments
# =====================================================================
class RechercheArgs(BaseModel):
    """Arguments de la recherche documentaire."""

    requete: str = Field(description="La question ou les mots-clés à rechercher")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Nombre d'extraits à remonter")


class CatalogueArgs(BaseModel):
    """Arguments de la consultation du catalogue de modèles."""

    actifs_seulement: bool = Field(default=True)


class EquipeArgs(BaseModel):
    code_team: str = Field(description="Code de l'équipe, par exemple DATA-01")


# =====================================================================
# Registre
# =====================================================================
class Outil(BaseModel):
    """Description d'un outil, indépendante de l'implémentation d'agent."""

    nom: str
    description: str
    schema_args: type[BaseModel]
    fonction: Callable[..., Awaitable[Any]]
    lecture_seule: bool = True

    model_config = {"arbitrary_types_allowed": True}

    # -----------------------------------------------------------------
    def schema_openai(self) -> dict[str, Any]:
        """Traduction au format attendu par l'API de complétion.

        Le schéma JSON est dérivé du modèle Pydantic : la description des
        paramètres n'est écrite qu'une fois, dans le modèle.
        """
        params = self.schema_args.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.nom,
                "description": self.description,
                "parameters": params,
            },
        }

    # -----------------------------------------------------------------
    async def appeler(self, arguments: dict[str, Any]) -> Any:
        """Valide puis exécute.

        La validation Pydantic est le garde-fou : un modèle qui invente
        un argument ou se trompe de type reçoit une erreur exploitable
        plutôt que de faire échouer la fonction au milieu.
        """
        try:
            valides = self.schema_args(**arguments)
        except ValidationError as exc:
            erreurs = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ValueError(f"Arguments invalides pour '{self.nom}' — {erreurs}") from exc

        resultat = self.fonction(**valides.model_dump())
        if inspect.isawaitable(resultat):
            resultat = await resultat
        return resultat


class RegistreOutils:
    """Collection d'outils, filtrable selon le contexte d'exécution."""

    def __init__(self, outils: list[Outil] | None = None):
        self._outils: dict[str, Outil] = {o.nom: o for o in (outils or [])}

    def ajouter(self, outil: Outil) -> None:
        if outil.nom in self._outils:
            raise ValueError(f"Un outil nommé '{outil.nom}' est déjà enregistré.")
        self._outils[outil.nom] = outil

    def get(self, nom: str) -> Outil | None:
        return self._outils.get(nom)

    def lister(self, lecture_seule_uniquement: bool = False) -> list[Outil]:
        """Les outils disponibles.

        `lecture_seule_uniquement` est le garde-fou du premier palier :
        tant qu'aucune validation humaine n'est en place, un agent ne
        doit pas pouvoir écrire.
        """
        outils = list(self._outils.values())
        if lecture_seule_uniquement:
            outils = [o for o in outils if o.lecture_seule]
        return sorted(outils, key=lambda o: o.nom)

    def schemas_openai(self, lecture_seule_uniquement: bool = True) -> list[dict[str, Any]]:
        return [o.schema_openai() for o in self.lister(lecture_seule_uniquement)]

    def __len__(self) -> int:
        return len(self._outils)

    def __contains__(self, nom: object) -> bool:
        return nom in self._outils


# =====================================================================
# Outils du projet
# =====================================================================
def outils_rag(rechercher: Callable[..., Awaitable[list[Source]]]) -> RegistreOutils:
    """Registre minimal branché sur le moteur de recherche documentaire.

    `rechercher(requete, top_k) -> list[Source]` est injecté : le package
    Agent ne dépend pas de l'implémentation du RAG, ce qui permet de le
    tester sans base vectorielle.
    """

    async def _rechercher(requete: str, top_k: int = 5) -> str:
        sources = await rechercher(requete=requete, top_k=top_k)
        if not sources:
            return "Aucun extrait pertinent trouvé."
        return "\n\n".join(
            f"[{i + 1}] {s.reference}\n{s.extrait}" for i, s in enumerate(sources)
        )

    return RegistreOutils([
        Outil(
            nom="recherche_documentaire",
            description=(
                "Recherche des extraits dans la base documentaire interne. "
                "À utiliser dès que la question porte sur des documents, "
                "procédures ou données propres à l'entreprise."
            ),
            schema_args=RechercheArgs,
            fonction=_rechercher,
            lecture_seule=True,
        ),
    ])
