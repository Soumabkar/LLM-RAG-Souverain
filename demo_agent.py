"""Démonstration des trois agents : python demo_agent.py

Fait tourner `LoopAgentic`, `PydanticAgent` et `LangGraphAgent` sur la
même question et le même registre d'outils, pour montrer qu'ils sont
interchangeables — puis exerce ce qui les distingue.

Prérequis : PostgreSQL démarré avec le schéma llm_souverain.
Ollama est facultatif : sans lui, le script bascule sur un client simulé
qui sait répondre aux appels d'outils. La plomberie est démontrée dans
les deux cas, seul le contenu des réponses change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import types

from Agent import AccesRefuse, BudgetEpuise, Source, outils_rag
from Engine.db import close_pool, get_connection
from Engine.models import model_llm, team, user
from LLM.llm import llm
from Model.ai_model import ai_model

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
for bruyant in ("Engine.models", "httpx", "httpx2", "openai", "openai._base_client",
                "Agent.loop_agentic", "Model.ai_model", "LLM.llm"):
    logging.getLogger(bruyant).setLevel(logging.WARNING)

# En mode simulé, l'inventaire du serveur d'inférence échoue à chaque
# appel : l'avertissement est attendu et noierait la démonstration.
logging.getLogger("LLM.llm").setLevel(logging.ERROR)

MODELE = os.getenv("LLM_MODEL", "llama3.2")
SECOND_MODELE = os.getenv("LLM_MODEL_2", "qwen2.5")
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

TABLES = (
    "llm_souverain.team_model, llm_souverain.model_llm, "
    "llm_souverain.team_member, llm_souverain.user_llm, llm_souverain.team_llm"
)

QUESTION = "Comment un utilisateur obtient-il l'accès à un modèle ?"


def titre(texte: str) -> None:
    print(f"\n{'─' * 72}\n  {texte}\n{'─' * 72}")


# =====================================================================
# Corpus documentaire — point de branchement avec le RAG du projet
# =====================================================================
CORPUS = [
    Source(
        reference="doc/securite.md#habilitations",
        extrait=(
            "L'accès aux modèles passe par l'équipe : un utilisateur hérite de "
            "l'union des modèles de toutes ses équipes, sans doublon."
        ),
        score=0.91,
    ),
    Source(
        reference="init/01_schema.sql#team_model",
        extrait="Table team_model : habilitation N-N entre une équipe et un modèle.",
        score=0.77,
    ),
]


async def rechercher(requete: str, top_k: int = 5) -> list[Source]:
    """À remplacer par un appel à la base vectorielle du projet.

        from RAG.retrieval import recherche_semantique
        resultats = await recherche_semantique(requete, top_k)
        return [Source(reference=r.chemin, extrait=r.texte, score=r.score)
                for r in resultats]
    """
    mots = {"accès", "acces", "modèle", "modele", "équipe", "equipe", "habilitation"}
    if any(m in requete.lower() for m in mots):
        return CORPUS[:top_k]
    return []


# =====================================================================
# Client simulé, utilisé quand Ollama est absent
# =====================================================================
class _ClientSimule:
    """Répond aux appels d'outils comme le ferait un vrai modèle.

    Premier tour : demande une recherche. Second tour : rédige la
    réponse à partir de l'extrait reçu.
    """

    base_url = OLLAMA

    def __init__(self) -> None:
        self.tours: dict[str, int] = {}

    @property
    def chat(self):
        return types.SimpleNamespace(completions=self)

    async def close(self) -> None:  # `ai_model.close()` la sollicite
        pass

    async def create(self, **kw):
        outils_dispo = bool(kw.get("tools"))
        cle = kw["model"]
        self.tours[cle] = self.tours.get(cle, 0) + 1
        a_deja_cherche = any(m.get("role") == "tool" for m in kw["messages"])

        if outils_dispo and not a_deja_cherche:
            appel = types.SimpleNamespace(
                id="call_1",
                function=types.SimpleNamespace(
                    name="recherche_documentaire",
                    arguments=json.dumps({"requete": QUESTION, "top_k": 2}),
                ),
            )
            message = types.SimpleNamespace(content=None, tool_calls=[appel])
        else:
            message = types.SimpleNamespace(
                content=(
                    f"[simulé · {kw['model']}] D'après [1], l'accès aux modèles passe "
                    "par l'équipe : un utilisateur hérite de l'union des modèles de "
                    "toutes ses équipes."
                ),
                tool_calls=None,
            )

        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            model=kw["model"],
            usage=types.SimpleNamespace(prompt_tokens=180, completion_tokens=42,
                                        total_tokens=222),
        )


def modele_pydantic_simule():
    """Modèle Pydantic AI simulé.

    Le client factice ne peut pas être branché derrière `OpenAIChatModel` :
    Pydantic AI attend un vrai `AsyncOpenAI`. On passe donc par
    `FunctionModel`, qui joue le même scénario — un appel d'outil, puis
    la sortie structurée.
    """
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def scenario(messages, info: AgentInfo) -> ModelResponse:
        a_cherche = any(
            part.__class__.__name__ == "ToolReturnPart"
            for message in messages for part in getattr(message, "parts", [])
        )
        if not a_cherche:
            return ModelResponse(parts=[ToolCallPart(
                "recherche_documentaire", {"requete": QUESTION, "top_k": 2}
            )])
        return ModelResponse(parts=[ToolCallPart("final_result", {
            "reponse": ("[simulé] D'après les extraits, l'accès aux modèles passe par "
                        "l'équipe : l'utilisateur hérite de l'union des modèles de "
                        "toutes ses équipes."),
            "sources": ["doc/securite.md#habilitations"],
            "confiance": 0.86,
        })])

    return FunctionModel(scenario, model_name="simule")


async def construire_client():
    """Ollama si joignable, client simulé sinon."""
    sonde = ai_model.from_ollama(model=MODELE, base_url=OLLAMA)
    if await sonde.health_check():
        print(f"Ollama joignable sur {OLLAMA} — modèles réels.")
        return sonde.client, True
    await sonde.close()
    print(f"Ollama injoignable sur {OLLAMA} — bascule sur un client simulé.")
    return _ClientSimule(), False


# =====================================================================
async def preparer(reel: bool) -> user:
    """Une équipe, deux modèles habilités, un troisième qui ne l'est pas."""
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(f"TRUNCATE {TABLES} CASCADE")

    await team("DATA-01", "Data Platform", "data@entreprise.fr").create_team()
    await team("SEC-01", "Cybersécurité").create_team()

    for code, nom in ((MODELE, MODELE), (SECOND_MODELE, SECOND_MODELE),
                      ("modele-interdit", "Modèle réservé")):
        await model_llm(code, nom).create_model(verify=False)

    await team("DATA-01", "").add_model(MODELE)
    await team("DATA-01", "").add_model(SECOND_MODELE)
    # « modele-interdit » n'est accordé qu'à SEC-01, dont karim n'est pas membre.
    await team("SEC-01", "").add_model("modele-interdit")

    karim = user("ksoumahoro", "MotDePasse!2026", "karim@entreprise.fr")
    await karim.create_user("DATA-01")

    habilites = [m["code_model"] for m in await karim.allowed_models()]
    print(f"Compte {karim.email} — équipes : {karim.teams} — modèles : {habilites}")
    return karim


async def nettoyer() -> None:
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(f"TRUNCATE {TABLES} CASCADE")
    await close_pool()


# =====================================================================
async def main() -> None:
    client, reel = await construire_client()
    karim = await preparer(reel)
    outils = outils_rag(rechercher)

    session = llm(ai_model(MODELE, client), karim)
    await session.authorize()

    # -----------------------------------------------------------------
    titre("1. Trois implémentations, une seule interface")
    print(f"Question : {QUESTION}\n")

    types_agents = ["loop"]
    for nom_module, type_agent in (("pydantic_ai", "pydantic"), ("langgraph", "langgraph")):
        try:
            __import__(nom_module)
            types_agents.append(type_agent)
        except ImportError:
            print(f"· {nom_module} absent — palier ignoré")

    for type_agent in types_agents:
        options = {}
        if type_agent == "pydantic" and not reel:
            options["modele_pydantic_ai"] = modele_pydantic_simule()
        agent = session.agent(type_agent, outils, **options)
        try:
            reponse = await agent.run(QUESTION)
        except (AccesRefuse, BudgetEpuise) as exc:
            print(f"  {agent.nom:<12} échec : {exc}")
            continue

        print(f"  {agent.nom:<12} {reponse.contenu[:110]}")
        print(f"  {'':<12} modèle={reponse.modele} · outils={reponse.outils_appeles} "
              f"· {reponse.iterations} itération(s) · {reponse.latence_ms} ms")

    # -----------------------------------------------------------------
    titre("2. L'agent suit le modèle choisi par l'utilisateur")
    agent = session.agent("loop", outils)
    print(f"  modèle de session : {await session.modele_courant()}")
    print(f"  exécution 1       : {(await agent.run(QUESTION)).modele}")

    await session.switch_model(SECOND_MODELE)
    print(f"  bascule vers      : {SECOND_MODELE}")
    print(f"  exécution 2       : {(await agent.run(QUESTION)).modele}"
          "   ← même instance d'agent, aucune reconstruction")

    # -----------------------------------------------------------------
    titre("3. Un modèle non habilité ferme l'accès à l'agent")

    # On force le modèle sur l'objet de session, comme le ferait un
    # appelant mal écrit : le contrôle de l'agent doit tenir quand même.
    session.model.model = "modele-interdit"
    try:
        await agent.run(QUESTION)
        print("  PROBLÈME : l'exécution aurait dû être refusée")
    except AccesRefuse as exc:
        print(f"  refus         : {exc}")

    print("\n  Révocation d'une habilitation en cours de session :")
    session.model.model = SECOND_MODELE
    await team("DATA-01", "").remove_model(SECOND_MODELE)
    try:
        await agent.run(QUESTION)
        print("  PROBLÈME : l'exécution aurait dû être refusée")
    except AccesRefuse as exc:
        print(f"  refus         : {str(exc)[:100]}…")
    await team("DATA-01", "").add_model(SECOND_MODELE)  # on rétablit

    # -----------------------------------------------------------------
    titre("4. Garde-fous de la boucle")
    await session.switch_model(MODELE)

    agent_lecture = session.agent("loop", outils, lecture_seule=True)
    reponse = await agent_lecture.run("Question sans rapport avec les documents")
    print(f"  question hors corpus : {reponse.iterations} itération(s), "
          f"outils={reponse.outils_appeles or 'aucun'}")

    print("  plafond d'itérations : BudgetEpuise est levé au-delà de max_iterations,")
    print("                         avec la liste des outils déjà appelés.")

    # -----------------------------------------------------------------
    if "langgraph" in types_agents:
        titre("5. LangGraph — reprise et validation humaine")
        graphe = session.agent("langgraph", outils, validation_humaine=True)

        avant = await graphe.run(QUESTION, thread_id="demo")
        etat = await graphe.etat_courant("demo")
        print(f"  après le contrôle qualité : {avant.contenu[:80]}")
        print(f"  en attente : {etat['en_attente']} · prochain nœud : {etat['prochain_noeud']}")
        print(f"  modèle mémorisé dans l'état : {etat['valeurs'].get('modele')}")

        apres = await graphe.reprendre("demo")
        print(f"\n  après approbation : {apres.contenu[:100]}")
        print(f"  parcours : {' → '.join(e.outil for e in apres.etapes)}")

        print("\n  Deux fils de conversation sont indépendants :")
        await graphe.run(QUESTION, thread_id="autre-fil")
        for fil in ("demo", "autre-fil"):
            etat = await graphe.etat_courant(fil)
            print(f"    {fil:<10} en attente={etat['en_attente']}")

    # -----------------------------------------------------------------
    titre("Récapitulatif")
    lignes = [
        ("Palier", "Dépendance", "Apport propre"),
        ("loop", "aucune", "outils, boucle, plafond d'itérations"),
        ("pydantic", "pydantic-ai-slim", "sortie structurée validée, observabilité"),
        ("langgraph", "langgraph", "état persistant, reprise, validation humaine"),
    ]
    for gauche, milieu, droite in lignes:
        print(f"  {gauche:<11} {milieu:<20} {droite}")
    print("\n  Les trois renvoient un ReponseAgent identique : basculer de l'un")
    print("  à l'autre est un changement de paramètre, pas une réécriture.")

    await session.model.close()
    await nettoyer()
    print("\nBase nettoyée, connexions fermées.")


if __name__ == "__main__":
    asyncio.run(main())
