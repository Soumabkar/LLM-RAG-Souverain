"""Tests du package Agent.

Aucun serveur d'inférence n'est requis : les trois implémentations sont
exercées avec des clients simulés ou le modèle de test de Pydantic AI.
"""

from __future__ import annotations

import json
import types

import pytest

from Agent import (
    BudgetEpuise,
    Etape,
    LoopAgentic,
    ReponseAgent,
    Source,
    TypeEtape,
    outils_rag,
)


# =====================================================================
# Doublures
# =====================================================================
async def rechercher_ok(requete: str, top_k: int = 5) -> list[Source]:
    return [Source(reference="doc/securite.md", extrait="L'accès passe par l'équipe.",
                   score=0.9)][:top_k]


async def rechercher_vide(requete: str, top_k: int = 5) -> list[Source]:
    return []


class AppelOutil:
    def __init__(self, nom: str, args: dict, identifiant: str = "call_1"):
        self.id = identifiant
        self.function = types.SimpleNamespace(name=nom, arguments=json.dumps(args))


class ClientSimule:
    """Rejoue un scénario de réponses : (contenu, appels d'outils)."""

    def __init__(self, scenario: list[tuple]):
        self.scenario = list(scenario)
        self.requetes: list[dict] = []
        self.modeles: list[str] = []

    @property
    def chat(self):
        return types.SimpleNamespace(completions=self)

    async def create(self, **kw):
        self.requetes.append(kw)
        self.modeles.append(kw["model"])
        contenu, appels = self.scenario.pop(0)
        message = types.SimpleNamespace(content=contenu, tool_calls=appels)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=types.SimpleNamespace(total_tokens=100),
        )


class SessionFactice:
    """Doublure de la classe `llm` : client, modèle courant, droits.

    Le protocole SessionLLM tient en trois membres, ce qui permet
    d'exercer les agents sans base ni serveur d'inférence.
    """

    def __init__(self, client, modele: str = "llama3.2",
                 autorises: set[str] | None = None,
                 email: str = "karim@entreprise.fr", technique: bool = False):
        self._client = client
        self._modele = modele
        self.autorises = set(autorises or {modele})
        self.user = types.SimpleNamespace(email=email, technique=technique)
        self.appels_autorise = 0

    @property
    def client(self):
        return self._client

    async def modele_courant(self) -> str:
        return self._modele

    async def autorise(self, code_model: str) -> bool:
        self.appels_autorise += 1
        return code_model in self.autorises

    def basculer(self, code: str) -> None:
        self._modele = code


def session(scenario, modele="llama3.2", autorises=None, **kw) -> SessionFactice:
    """Raccourci : un client simulé et sa session."""
    return SessionFactice(ClientSimule(scenario), modele, autorises, **kw)


@pytest.fixture
def registre():
    return outils_rag(rechercher_ok)


# =====================================================================
class TestRegistre:
    def test_schema_derive_du_modele(self, registre):
        schema = registre.schemas_openai()[0]
        assert schema["function"]["name"] == "recherche_documentaire"
        props = schema["function"]["parameters"]["properties"]
        assert "requete" in props and "top_k" in props
        assert props["top_k"]["maximum"] == 20

    async def test_validation_des_arguments(self, registre):
        outil = registre.get("recherche_documentaire")
        with pytest.raises(ValueError, match="top_k"):
            await outil.appeler({"requete": "x", "top_k": 99})

    async def test_argument_manquant(self, registre):
        outil = registre.get("recherche_documentaire")
        with pytest.raises(ValueError, match="requete"):
            await outil.appeler({})

    def test_doublon_refuse(self, registre):
        outil = registre.get("recherche_documentaire")
        with pytest.raises(ValueError, match="déjà enregistré"):
            registre.ajouter(outil)

    def test_filtre_lecture_seule(self, registre):
        from Agent import Outil
        from Agent.tools import RechercheArgs

        async def ecrire(requete: str, top_k: int = 5) -> str:
            return "écrit"

        registre.ajouter(Outil(nom="ecrire", description="écrit",
                               schema_args=RechercheArgs, fonction=ecrire,
                               lecture_seule=False))
        assert len(registre.lister()) == 2
        assert len(registre.lister(lecture_seule_uniquement=True)) == 1


# =====================================================================
class TestLoopAgentic:
    async def test_appel_outil_puis_reponse(self, registre):
        sess = session([
            (None, [AppelOutil("recherche_documentaire", {"requete": "habilitation"})]),
            ("D'après [1], l'accès passe par l'équipe.", None),
        ])
        reponse = await LoopAgentic(sess, registre).run("Question ?")

        assert isinstance(reponse, ReponseAgent)
        assert reponse.outils_appeles == ["recherche_documentaire"]
        assert reponse.iterations == 2
        assert reponse.tokens == 200

    async def test_reponse_directe_sans_outil(self, registre):
        sess = session([("Bonjour.", None)])
        reponse = await LoopAgentic(sess, registre).run("Bonjour")
        assert reponse.outils_appeles == []
        assert reponse.iterations == 1

    async def test_erreur_de_validation_renvoyee_au_modele(self, registre):
        """Un argument invalide doit permettre au modèle de se corriger,
        pas faire échouer toute la requête."""
        sess = session([
            (None, [AppelOutil("recherche_documentaire", {"requete": "x", "top_k": 99})]),
            ("Corrigé.", None),
        ])
        reponse = await LoopAgentic(sess, registre).run("Question ?")

        assert reponse.etapes[0].type is TypeEtape.ERREUR
        assert "top_k" in reponse.etapes[0].observation
        assert reponse.contenu == "Corrigé."
        # Le retour d'erreur doit être présent dans le contexte du 2e appel
        messages = sess.client.requetes[1]["messages"]
        assert any(m.get("role") == "tool" for m in messages)

    async def test_outil_inconnu(self, registre):
        sess = session([
            (None, [AppelOutil("supprimer_tout", {})]),
            ("Impossible.", None),
        ])
        reponse = await LoopAgentic(sess, registre).run("Question ?")
        assert "inconnu" in reponse.etapes[0].observation

    async def test_outil_en_ecriture_refuse_en_lecture_seule(self, registre):
        from Agent import Outil
        from Agent.tools import RechercheArgs

        async def ecrire(requete: str, top_k: int = 5) -> str:
            raise AssertionError("ne doit pas être appelé")

        registre.ajouter(Outil(nom="ecrire", description="écrit",
                               schema_args=RechercheArgs, fonction=ecrire,
                               lecture_seule=False))
        sess = session([
            (None, [AppelOutil("ecrire", {"requete": "x"})]),
            ("Impossible.", None),
        ])
        reponse = await LoopAgentic(sess, registre,
                                    lecture_seule=True).run("Question ?")
        assert "lecture seule" in reponse.etapes[0].observation

    async def test_budget_epuise(self, registre):
        boucle = [(None, [AppelOutil("recherche_documentaire", {"requete": "x"})])] * 4
        with pytest.raises(BudgetEpuise, match="3 tours"):
            await LoopAgentic(SessionFactice(ClientSimule(boucle)), registre,
                              max_iterations=3).run("Question ?")

    async def test_json_illisible(self, registre):
        appel = AppelOutil("recherche_documentaire", {})
        appel.function.arguments = "{ceci n'est pas du json"
        sess = session([(None, [appel]), ("Corrigé.", None)])
        reponse = await LoopAgentic(sess, registre).run("Question ?")
        assert reponse.etapes[0].type is TypeEtape.ERREUR

    async def test_les_outils_sont_transmis(self, registre):
        sess = session([("Bonjour.", None)])
        await LoopAgentic(sess, registre).run("Bonjour")
        assert sess.client.requetes[0]["tools"][0]["function"]["name"] == "recherche_documentaire"
        assert sess.client.requetes[0]["tool_choice"] == "auto"


# =====================================================================
class TestEtape:
    def test_observation_tronquee(self):
        """La trace ne doit pas embarquer des milliers de lignes."""
        etape = Etape(numero=1, type=TypeEtape.OUTIL, observation="x" * 5000)
        assert len(etape.observation) < 2100
        assert "tronqués" in etape.observation

    def test_numero_positif(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Etape(numero=0, type=TypeEtape.REPONSE)


# =====================================================================
class TestPydanticAgent:
    async def test_sortie_structuree(self, registre):
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        from Agent.pydantic_agent import PydanticAgent

        agent = PydanticAgent(SessionFactice(None), registre,
                              modele_pydantic_ai=TestModel())
        reponse = await agent.run("Question ?")

        assert isinstance(reponse, ReponseAgent)
        # L'outil interne de sortie structurée ne doit pas polluer la trace
        assert reponse.outils_appeles == ["recherche_documentaire"]

    async def test_meme_interface_que_loop(self, registre):
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        from Agent.pydantic_agent import PydanticAgent

        agents = [
            LoopAgentic(session([("Réponse.", None)]), registre),
            PydanticAgent(SessionFactice(None), registre, modele_pydantic_ai=TestModel()),
        ]
        for agent in agents:
            reponse = await agent.run("Question ?")
            assert isinstance(reponse, ReponseAgent)
            assert reponse.modele == "llama3.2"


# =====================================================================
class TestLangGraphAgent:
    @staticmethod
    def _client():
        class Client:
            @property
            def chat(self):
                return types.SimpleNamespace(completions=self)

            async def create(self, **kw):
                return types.SimpleNamespace(choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="D'après [1], réponse."))])

        return Client()

    async def test_parcours_nominal(self, registre):
        pytest.importorskip("langgraph")
        from Agent.langgraph_agent import LangGraphAgent

        agent = LangGraphAgent(SessionFactice(self._client()), registre)
        reponse = await agent.run("Question ?", thread_id="nominal")

        assert [e.outil for e in reponse.etapes] == [
            "analyser", "rechercher", "controler", "generer"
        ]

    async def test_abandon_propre_sans_extrait(self):
        pytest.importorskip("langgraph")
        from Agent.langgraph_agent import LangGraphAgent

        agent = LangGraphAgent(SessionFactice(self._client()), outils_rag(rechercher_vide),
                               max_iterations=2)
        reponse = await agent.run("Question ?", thread_id="vide")

        assert "Aucun extrait" in reponse.contenu
        assert reponse.iterations == 2

    async def test_interruption_puis_reprise(self, registre):
        """La validation humaine suspend le graphe, l'état est conservé,
        et la reprise ne rejoue pas les étapes déjà faites."""
        pytest.importorskip("langgraph")
        from Agent.langgraph_agent import LangGraphAgent

        agent = LangGraphAgent(SessionFactice(self._client()), registre,
                               validation_humaine=True)
        avant = await agent.run("Question ?", thread_id="validation")
        assert "validation humaine" in avant.contenu

        etat = await agent.etat_courant("validation")
        assert etat["en_attente"] is True

        apres = await agent.reprendre("validation")
        assert "réponse" in apres.contenu.lower()
        assert (await agent.etat_courant("validation"))["en_attente"] is False

    async def test_fils_isoles(self, registre):
        pytest.importorskip("langgraph")
        from Agent.langgraph_agent import LangGraphAgent

        agent = LangGraphAgent(SessionFactice(self._client()), registre)
        await agent.run("Première question ?", thread_id="fil-a")
        await agent.run("Seconde question ?", thread_id="fil-b")

        a = await agent.etat_courant("fil-a")
        b = await agent.etat_courant("fil-b")
        assert a["valeurs"]["question"] != b["valeurs"]["question"]


# =====================================================================
class TestModeleCourant:
    """L'agent suit le modèle de l'utilisateur, et contrôle ses droits."""

    async def test_suit_la_bascule_sans_reconstruire_l_agent(self, registre):
        sess = session([("ok", None), ("ok", None)],
                       autorises={"llama3.2", "qwen2.5"})
        agent = LoopAgentic(sess, registre)

        premier = await agent.run("Question 1")
        sess.basculer("qwen2.5")
        second = await agent.run("Question 2")

        assert premier.modele == "llama3.2"
        assert second.modele == "qwen2.5"
        assert sess.client.modeles == ["llama3.2", "qwen2.5"]

    async def test_modele_non_habilite_refuse(self, registre):
        from Agent import AccesRefuse

        agent = LoopAgentic(session([("ok", None)], "mistral", {"llama3.2"}), registre)

        with pytest.raises(AccesRefuse, match="mistral"):
            await agent.run("Question ?")

    async def test_aucun_appel_au_modele_si_refus(self, registre):
        """Le contrôle précède l'appel : pas de jeton consommé."""
        from Agent import AccesRefuse

        sess = session([("ok", None)], "mistral", {"llama3.2"})
        agent = LoopAgentic(sess, registre)

        with pytest.raises(AccesRefuse):
            await agent.run("Question ?")
        assert sess.client.modeles == []

    async def test_revocation_en_cours_de_session(self, registre):
        """Le droit est revérifié à chaque exécution, jamais mis en cache."""
        from Agent import AccesRefuse

        sess = session([("ok", None), ("ok", None)])
        agent = LoopAgentic(sess, registre)

        assert (await agent.run("Question 1")).modele == "llama3.2"
        sess.autorises.discard("llama3.2")
        with pytest.raises(AccesRefuse):
            await agent.run("Question 2")
        assert sess.appels_autorise == 2

    async def test_compte_technique(self, registre):
        """Un agent sous compte de service ne diffère que par sa session."""
        sess = session([("ok", None)], email="agent-rag@interne", technique=True)
        agent = LoopAgentic(sess, registre)

        reponse = await agent.run("Question ?")
        assert reponse.modele == "llama3.2"
        assert agent.titulaire == "agent-rag@interne"

    async def test_titulaire_dans_le_message_de_refus(self, registre):
        """Le refus doit nommer le compte concerné : avec des comptes
        techniques, « cet utilisateur » ne suffit plus à diagnostiquer."""
        from Agent import AccesRefuse

        sess = session([("ok", None)], "mistral", {"llama3.2"},
                       email="agent-rag@interne", technique=True)
        with pytest.raises(AccesRefuse, match="agent-rag@interne"):
            await LoopAgentic(sess, registre).run("Question ?")

    async def test_pydantic_agent_suit_aussi(self, registre):
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        from Agent.pydantic_agent import PydanticAgent

        sess = SessionFactice(None, autorises={"llama3.2", "qwen2.5"})
        agent = PydanticAgent(sess, registre, modele_pydantic_ai=TestModel())

        assert (await agent.run("Q1")).modele == "llama3.2"
        sess.basculer("qwen2.5")
        assert (await agent.run("Q2")).modele == "qwen2.5"

    async def test_pydantic_agent_refuse(self, registre):
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        from Agent import AccesRefuse
        from Agent.pydantic_agent import PydanticAgent

        sess = SessionFactice(None, "mistral", {"llama3.2"})
        agent = PydanticAgent(sess, registre, modele_pydantic_ai=TestModel())
        with pytest.raises(AccesRefuse):
            await agent.run("Question ?")

    async def test_langgraph_enregistre_le_modele_dans_l_etat(self, registre):
        """Le modèle transite par l'état : une reprise après incident
        réutilise celui qui avait été autorisé au départ."""
        pytest.importorskip("langgraph")
        from Agent.langgraph_agent import LangGraphAgent

        agent = LangGraphAgent(SessionFactice(TestLangGraphAgent._client()), registre)

        reponse = await agent.run("Question ?", thread_id="modele")
        assert reponse.modele == "llama3.2"
        etat = await agent.etat_courant("modele")
        assert etat["valeurs"]["modele"] == "llama3.2"

    async def test_langgraph_refuse(self, registre):
        pytest.importorskip("langgraph")
        from Agent import AccesRefuse
        from Agent.langgraph_agent import LangGraphAgent

        sess = SessionFactice(TestLangGraphAgent._client(), "mistral", {"llama3.2"})
        agent = LangGraphAgent(sess, registre)
        with pytest.raises(AccesRefuse):
            await agent.run("Question ?", thread_id="refus")

    async def test_langgraph_reverifie_a_la_reprise(self, registre):
        """Une validation humaine peut intervenir des heures plus tard :
        les droits ont pu changer entre-temps."""
        pytest.importorskip("langgraph")
        from Agent import AccesRefuse
        from Agent.langgraph_agent import LangGraphAgent

        sess = SessionFactice(TestLangGraphAgent._client())
        agent = LangGraphAgent(sess, registre, validation_humaine=True)
        await agent.run("Question ?", thread_id="reprise-droits")

        sess.autorises.discard("llama3.2")
        with pytest.raises(AccesRefuse):
            await agent.reprendre("reprise-droits")
