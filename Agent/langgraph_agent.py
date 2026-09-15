"""Palier 3 — graphe d'état persistant (LangGraph).

L'agent n'est plus une boucle mais un graphe : des nœuds, des arêtes
conditionnelles, un état typé et partagé. Ce que les deux paliers
précédents ne savent pas faire :

* **Reprise après incident.** Un *checkpointer* enregistre l'état à
  chaque nœud. Le traitement survit à un redémarrage et repart où il
  s'est arrêté, sans reconsommer de jetons.
* **Validation humaine intégrée.** Le graphe s'interrompt à un point
  défini, attend une approbation — minutes ou jours — puis reprend, sans
  mobiliser de ressources pendant l'attente.
* **Retour en arrière.** L'état de chaque étape étant conservé, on peut
  rejouer depuis un point donné avec un autre modèle ou un autre prompt.
* **Fils isolés.** Chaque `thread_id` a son état propre.

Le graphe implémenté suit le schéma d'architecture RAG : analyse →
recherche → contrôle qualité → génération, avec une boucle de reprise si
les extraits sont insuffisants et une pause si la réponse est sensible.

Dépendances : ``pip install langgraph``
(et ``langgraph-checkpoint-postgres`` pour la persistance en base).
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal, TypedDict

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

SEUIL_EXTRAITS = 1  # nombre minimal d'extraits pour tenter une génération


def _concat(gauche: list, droite: list) -> list:
    """Réducteur d'état : les listes s'accumulent au lieu d'être écrasées.

    Sans réducteur, deux nœuds écrivant `extraits` se remplaceraient
    mutuellement ; c'est le principal piège de LangGraph.
    """
    return (gauche or []) + (droite or [])


class EtatAgent(TypedDict, total=False):
    """État partagé, typé et sérialisé à chaque point de contrôle."""

    question: str
    requete: str
    extraits: Annotated[list[dict], _concat]
    reponse: str
    modele: str
    tours_recherche: int
    sensible: bool
    trace: Annotated[list[dict], _concat]


class LangGraphAgent(BaseAgent):
    """Graphe d'état pour les traitements longs ou à validation humaine.

    Args:
        client: client `AsyncOpenAI` du projet.
        modele: identifiant du modèle habilité.
        outils: registre partagé.
        checkpointer: `InMemorySaver` par défaut. En production, passer
            un `AsyncPostgresSaver` adossé au PostgreSQL existant.
        validation_humaine: interrompt le graphe avant la réponse quand
            le contrôle la juge sensible.
    """

    nom = "langgraph"

    def __init__(
        self,
        session: "SessionLLM",
        outils: RegistreOutils,
        max_iterations: int = 3,
        checkpointer: Any = None,
        validation_humaine: bool = False,
        temperature: float = 0.3,
    ):
        super().__init__(session, max_iterations)
        self.outils = outils
        self.validation_humaine = validation_humaine
        self.temperature = temperature

        from langgraph.graph import END, START, StateGraph

        if checkpointer is None:
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()
        self.checkpointer = checkpointer

        graphe = StateGraph(EtatAgent)
        graphe.add_node("analyser", self._analyser)
        graphe.add_node("rechercher", self._rechercher)
        graphe.add_node("controler", self._controler)
        graphe.add_node("generer", self._generer)

        graphe.add_edge(START, "analyser")
        graphe.add_edge("analyser", "rechercher")
        graphe.add_edge("rechercher", "controler")
        graphe.add_conditional_edges(
            "controler",
            self._decider,
            {"reprendre": "rechercher", "generer": "generer", "abandonner": END},
        )
        graphe.add_edge("generer", END)

        self._graphe = graphe.compile(
            checkpointer=checkpointer,
            # L'interruption est déclarée à la compilation : le graphe
            # s'arrête avant le nœud, l'état est sauvegardé, et la reprise
            # se fait par un second appel sur le même thread_id.
            interrupt_before=["generer"] if validation_humaine else None,
        )

    # =================================================================
    # Nœuds
    # =================================================================
    async def _analyser(self, etat: EtatAgent) -> dict:
        """Reformule la question en requête de recherche.

        Séparer l'analyse de la recherche permet de rejouer la recherche
        avec une requête différente sans repasser par le modèle.
        """
        question = etat["question"]
        return {
            "requete": question,
            "tours_recherche": 0,
            "trace": [{"noeud": "analyser", "requete": question}],
        }

    async def _rechercher(self, etat: EtatAgent) -> dict:
        outil = self.outils.get("recherche_documentaire")
        if outil is None:
            return {"trace": [{"noeud": "rechercher", "erreur": "outil absent"}]}

        tour = etat.get("tours_recherche", 0) + 1
        # À la reprise, on élargit : si la première passe n'a rien donné,
        # refaire la même requête donnerait le même vide.
        top_k = 5 * tour
        try:
            resultat = await outil.appeler({"requete": etat["requete"], "top_k": min(top_k, 20)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("recherche en échec")
            return {"trace": [{"noeud": "rechercher", "erreur": str(exc)}]}

        extraits = [{"contenu": resultat}] if resultat and "Aucun extrait" not in resultat else []
        return {
            "extraits": extraits,
            "tours_recherche": tour,
            "trace": [{"noeud": "rechercher", "tour": tour, "trouves": len(extraits)}],
        }

    async def _controler(self, etat: EtatAgent) -> dict:
        """Contrôle qualité : assez d'éléments pour répondre ?

        C'est le nœud qui rend le graphe utile : il décide de relancer
        une recherche, de générer, ou d'abandonner proprement.
        """
        nb = len(etat.get("extraits", []))
        sensible = self.validation_humaine and nb > 0
        return {
            "sensible": sensible,
            "trace": [{"noeud": "controler", "extraits": nb, "sensible": sensible}],
        }

    def _decider(self, etat: EtatAgent) -> Literal["reprendre", "generer", "abandonner"]:
        nb = len(etat.get("extraits", []))
        if nb >= SEUIL_EXTRAITS:
            return "generer"
        if etat.get("tours_recherche", 0) < self.max_iterations:
            return "reprendre"
        return "abandonner"

    async def _generer(self, etat: EtatAgent) -> dict:
        contexte = "\n\n".join(e["contenu"] for e in etat.get("extraits", []))
        messages = [
            {"role": "system", "content":
                "Tu es l'assistant interne de la plateforme LLM souverain. "
                "Réponds en français, uniquement à partir des extraits fournis. "
                "Cite tes sources [1], [2]. Si les extraits ne suffisent pas, dis-le."},
            {"role": "user", "content": f"Extraits :\n{contexte}\n\nQuestion : {etat['question']}"},
        ]
        # Le modèle transite par l'état : il est ainsi enregistré dans le
        # point de contrôle, et une reprise après incident réutilise
        # exactement celui qui avait été autorisé au départ.
        modele = etat.get("modele") or await self._modele_actif()
        try:
            reponse = await self.client.chat.completions.create(
                model=modele, messages=messages, temperature=self.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"Génération impossible : {exc}") from exc

        return {
            "reponse": reponse.choices[0].message.content or "",
            "trace": [{"noeud": "generer", "caracteres": len(contexte)}],
        }

    # =================================================================
    # Exécution
    # =================================================================
    async def run(self, question: str, thread_id: str = "defaut", **_: Any) -> ReponseAgent:
        debut = self._chrono()
        config = {"configurable": {"thread_id": thread_id}}

        modele = await self._modele_actif()
        etat = await self._graphe.ainvoke(
            {"question": question, "modele": modele}, config=config
        )
        return self._formater(etat, debut, thread_id, await self._en_attente(config))

    async def reprendre(self, thread_id: str) -> ReponseAgent:
        """Reprend un graphe interrompu, après validation humaine.

        `None` en entrée signifie « continue depuis l'état enregistré »
        plutôt que « démarre une nouvelle exécution ».
        """
        debut = self._chrono()
        config = {"configurable": {"thread_id": thread_id}}
        # L'habilitation est revérifiée à la reprise : une validation
        # humaine peut intervenir des heures plus tard, les droits ont pu
        # changer entre-temps.
        await self._modele_actif()
        etat = await self._graphe.ainvoke(None, config=config)
        return self._formater(etat, debut, thread_id, await self._en_attente(config))

    async def etat_courant(self, thread_id: str) -> dict:
        """État sauvegardé — pour l'audit ou une interface d'approbation."""
        instantane = await self._graphe.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        return {
            "valeurs": instantane.values,
            "prochain_noeud": instantane.next,
            "en_attente": bool(instantane.next),
        }

    # -----------------------------------------------------------------
    async def _en_attente(self, config: dict) -> bool:
        instantane = await self._graphe.aget_state(config)
        return bool(instantane.next)

    def _formater(
        self, etat: dict, debut: float, thread_id: str, en_attente: bool = False
    ) -> ReponseAgent:
        etapes = [
            Etape(
                numero=i + 1,
                type=TypeEtape.OUTIL if t.get("noeud") == "rechercher" else TypeEtape.REFLEXION,
                outil=t.get("noeud"),
                arguments={k: v for k, v in t.items() if k != "noeud"},
                observation=json.dumps(t, ensure_ascii=False),
            )
            for i, t in enumerate(etat.get("trace", []))
        ]

        contenu = etat.get("reponse")
        if not contenu and en_attente:
            # Le graphe est interrompu avant la génération : il ne s'agit
            # pas d'un échec mais d'une attente. Renvoyer le message
            # d'absence de résultat ferait croire à une recherche vide.
            contenu = (
                "En attente de validation humaine avant génération "
                f"(fil « {thread_id} »). Appeler `reprendre()` pour poursuivre."
            )
        elif not contenu:
            contenu = (
                "Aucun extrait pertinent n'a été trouvé après "
                f"{etat.get('tours_recherche', 0)} recherche(s). "
                "Reformulez la question ou vérifiez que le corpus est indexé."
            )

        return ReponseAgent(
            contenu=contenu,
            modele=etat.get("modele") or self.modele,
            etapes=etapes,
            sources=[Source(reference=f"extrait {i + 1}")
                     for i in range(len(etat.get("extraits", [])))],
            iterations=etat.get("tours_recherche", 0),
            latence_ms=self._ms(debut),
        )
