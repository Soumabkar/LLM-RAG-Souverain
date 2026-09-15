"""Catalogue de modèles et habilitations par équipe."""

from __future__ import annotations

import pytest

from Engine.models import model_llm, team, user

pytestmark = pytest.mark.db


# =====================================================================
class TestCatalogue:
    async def test_ajout_nominal(self, db):
        res = await model_llm("llama3.1:8b", "Llama 3.1 8B", "Généraliste").create_model(verify=False)
        assert res
        assert [m["code_model"] for m in await model_llm.catalogue()] == ["llama3.1:8b"]

    async def test_doublon_refuse(self, catalogue):
        res = await model_llm("llama3.1:8b", "Autre nom").create_model(verify=False)
        assert not res
        assert "existe déjà" in res.message

    async def test_code_obligatoire(self, db):
        res = await model_llm("", "Nom").create_model(verify=False)
        assert not res
        assert "obligatoire" in res.message

    async def test_code_trop_long(self, db):
        res = await model_llm("m" * 200, "Nom").create_model(verify=False)
        assert not res
        assert "200" in res.message

    async def test_catalogue_trie_par_nom_affiche(self, catalogue):
        noms = [m["display_name"] for m in await model_llm.catalogue()]
        assert noms == sorted(noms)

    async def test_suppression_indique_les_habilitations(self, catalogue, equipe_data):
        await equipe_data.add_model("mistral:7b")
        res = await model_llm("mistral:7b", "Mistral 7B").delete_model()
        assert res
        assert "1 habilitation" in res.message

    async def test_modele_absent(self, db):
        res = await model_llm("fantome:1b", "Fantôme").delete_model()
        assert not res
        assert "n'existe pas" in res.message


# =====================================================================
class TestHabilitationEquipe:
    async def test_autorisation_nominale(self, catalogue, equipe_data):
        assert await equipe_data.add_model("llama3.1:8b")
        assert [m["code_model"] for m in await equipe_data.load_models()] == ["llama3.1:8b"]

    async def test_doublon_refuse(self, catalogue, equipe_data):
        await equipe_data.add_model("llama3.1:8b")
        res = await equipe_data.add_model("llama3.1:8b")
        assert not res
        assert "déjà autorisé" in res.message

    async def test_modele_hors_catalogue(self, catalogue, equipe_data):
        res = await equipe_data.add_model("inconnu:1b")
        assert not res
        assert "catalogue" in res.message

    async def test_equipe_inexistante(self, catalogue):
        res = await team("FANTOME", "").add_model("llama3.1:8b")
        assert not res
        assert "n'existe pas" in res.message

    async def test_retrait(self, catalogue, equipe_data):
        await equipe_data.add_model("llama3.1:8b")
        assert await equipe_data.remove_model("llama3.1:8b")
        assert await equipe_data.load_models() == []

    async def test_retrait_non_autorise(self, catalogue, equipe_data):
        res = await equipe_data.remove_model("llama3.1:8b")
        assert not res
        assert "n'est pas autorisé" in res.message

    async def test_suppression_equipe_retire_les_habilitations(self, catalogue, equipe_data, sql):
        await equipe_data.add_model("llama3.1:8b")
        await equipe_data.delete_team()
        lignes = await sql("SELECT * FROM llm_souverain.team_model")
        assert lignes == []

    async def test_le_modele_survit_a_la_suppression_de_l_equipe(self, catalogue, equipe_data):
        await equipe_data.add_model("llama3.1:8b")
        await equipe_data.delete_team()
        assert len(await model_llm.catalogue()) == 3


# =====================================================================
class TestAccesUtilisateur:
    async def test_union_des_equipes(self, catalogue, equipe_data, equipe_secu, karim):
        await karim.join_team("SEC-01")
        await equipe_data.add_model("llama3.1:8b")
        await equipe_data.add_model("qwen2.5-coder:7b")
        await equipe_secu.add_model("mistral:7b")

        codes = [m["code_model"] for m in await karim.allowed_models()]
        assert codes == ["llama3.1:8b", "mistral:7b", "qwen2.5-coder:7b"]

    async def test_modele_commun_compte_une_fois(self, catalogue, equipe_data, equipe_secu, karim):
        await karim.join_team("SEC-01")
        await equipe_data.add_model("llama3.1:8b")
        await equipe_secu.add_model("llama3.1:8b")
        assert len(await karim.allowed_models()) == 1

    async def test_sans_equipe_aucun_modele(self, catalogue, db):
        seul = user("seul", "Secret!2026", "seul@entreprise.fr")
        await seul.create_user()
        assert await seul.allowed_models() == []

    async def test_can_use_model(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        assert await karim.can_use_model("llama3.1:8b")
        assert not await karim.can_use_model("mistral:7b")

    async def test_perte_d_acces_en_quittant_l_equipe(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        assert await karim.can_use_model("llama3.1:8b")
        await equipe_data.delete_member_team(karim)
        assert not await karim.can_use_model("llama3.1:8b")

    async def test_desactivation_globale(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        await model_llm("llama3.1:8b", "Llama 3.1 8B").set_active(False)
        assert await karim.allowed_models() == []
        assert not await karim.can_use_model("llama3.1:8b")

    async def test_reactivation_retrouve_l_habilitation(self, catalogue, equipe_data, karim):
        """La désactivation ne supprime pas l'habilitation, elle la masque."""
        await equipe_data.add_model("llama3.1:8b")
        await model_llm("llama3.1:8b", "Llama 3.1 8B").set_active(False)
        await model_llm("llama3.1:8b", "Llama 3.1 8B").set_active(True)
        assert await karim.can_use_model("llama3.1:8b")

    async def test_modele_inconnu_refuse(self, catalogue, equipe_data, karim):
        assert not await karim.can_use_model("nexiste:pas")


# =====================================================================
class TestModeleInstalle:
    """Contrôle d'installation avant mise au catalogue.

    Le vérificateur est injecté : les tests ne dépendent d'aucun serveur
    d'inférence.
    """

    INSTALLES = {"llama3.1:8b", "mistral:7b"}

    async def _installe(self, code: str) -> bool:
        return code in self.INSTALLES

    async def _serveur_eteint(self, code: str) -> bool:
        from Model.ai_model import AIModelError

        raise AIModelError("Serveur d'inférence injoignable sur http://localhost:11434/v1")

    async def test_modele_installe_accepte(self, db):
        res = await model_llm("llama3.1:8b", "Llama 3.1 8B").create_model(
            verifier=self._installe
        )
        assert res

    async def test_modele_absent_refuse(self, db):
        res = await model_llm("gpt-oss:120b", "GPT-OSS").create_model(verifier=self._installe)
        assert not res
        assert "n'est pas installé" in res.message
        assert "ollama pull" in res.message

    async def test_refus_n_ecrit_rien(self, db):
        await model_llm("gpt-oss:120b", "GPT-OSS").create_model(verifier=self._installe)
        assert await model_llm.catalogue() == []

    async def test_serveur_injoignable_distingue_du_modele_absent(self, db):
        """Deux causes, deux messages : sinon l'administrateur cherche au
        mauvais endroit."""
        res = await model_llm("llama3.1:8b", "Llama 3.1 8B").create_model(
            verifier=self._serveur_eteint
        )
        assert not res
        assert "injoignable" in res.message
        assert "n'est pas installé" not in res.message

    async def test_verify_false_contourne_le_controle(self, db):
        res = await model_llm("jamais-installe:1b", "Hors ligne").create_model(verify=False)
        assert res

    async def test_validation_avant_verification(self, db):
        """Un champ vide est refusé sans appeler le serveur."""

        async def interdit(code: str) -> bool:
            raise AssertionError("le vérificateur ne doit pas être appelé")

        res = await model_llm("", "Nom").create_model(verifier=interdit)
        assert not res
        assert "obligatoire" in res.message


# =====================================================================
class TestCorrespondanceTag:
    """Catalogage, supervision et sélecteur doivent appliquer la même
    règle : un code accepté à l'ajout ne doit pas être signalé manquant
    par la supervision juste après."""

    PARC = {"qwen2.5:latest", "llama3.2:latest", "nomic-embed-text:latest"}

    def test_correspondance_exacte(self):
        from Model.ai_model import matches_installed

        assert matches_installed("llama3.2:latest", self.PARC)

    def test_code_sans_tag_accepte_n_importe_quel_tag(self):
        from Model.ai_model import matches_installed

        assert matches_installed("qwen2.5", self.PARC)

    def test_tag_explicite_different_refuse(self):
        """llama3.2:3b et llama3.2:latest sont des poids différents."""
        from Model.ai_model import matches_installed

        assert not matches_installed("llama3.2:3b", self.PARC)

    def test_famille_inconnue_refusee(self):
        from Model.ai_model import matches_installed

        assert not matches_installed("gpt-oss", self.PARC)

    async def test_add_et_check_coherents(self, db, monkeypatch):
        import Model.ai_model as aim

        async def parc(base_url=None):
            return set(self.PARC)

        async def installe(code, base_url=None):
            return aim.matches_installed(code, self.PARC)

        monkeypatch.setattr(aim, "installed_models", parc)
        monkeypatch.setattr(aim, "is_model_installed", installe)

        assert await model_llm("qwen2.5", "Qwen 2.5").create_model()
        assert await model_llm.verifier_catalogue() == []


# =====================================================================
class TestTeamsDetail:
    """Liste des équipes affichée à l'utilisateur."""

    async def test_nom_et_modeles(self, catalogue, equipe_data, equipe_secu, karim):
        await karim.join_team("SEC-01")
        await equipe_data.add_model("llama3.1:8b")
        await equipe_data.add_model("qwen2.5-coder:7b")
        await equipe_secu.add_model("mistral:7b")

        detail = await karim.teams_detail()
        assert [e["code_team"] for e in detail] == ["SEC-01", "DATA-01"]  # tri par nom
        data = next(e for e in detail if e["code_team"] == "DATA-01")
        assert data["team_name"] == "Data Platform"
        assert sorted(data["models"]) == ["Llama 3.1 8B", "Qwen Coder"]

    async def test_equipe_sans_modele_reste_affichee(self, catalogue, equipe_data, karim):
        """C'est souvent l'explication d'une liste de modèles plus courte
        que prévu : la masquer rendrait le diagnostic impossible."""
        detail = await karim.teams_detail()
        assert [e["code_team"] for e in detail] == ["DATA-01"]
        assert detail[0]["models"] == []

    async def test_modele_desactive_exclu(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        await model_llm("llama3.1:8b", "Llama 3.1 8B").set_active(False)
        detail = await karim.teams_detail()
        assert detail[0]["models"] == []

    async def test_sans_equipe(self, catalogue, db):
        seul = user("seul", "Secret!2026", "seul@entreprise.fr")
        await seul.create_user()
        assert await seul.teams_detail() == []

    async def test_contact_present(self, catalogue, equipe_data, karim):
        assert (await karim.teams_detail())[0]["contact"] == "data@entreprise.fr"


# =====================================================================
class TestComptesTechniques:
    """Un agent tourne sous un compte de service, jamais sous un compte
    humain : ses actions doivent rester distinguables dans les journaux."""

    async def test_creation(self, db, equipe_data):
        compte, mdp = await user.creer_compte_technique(
            "agent-rag", "agent-rag@interne", "Agent RAG documentaire", "DATA-01"
        )
        assert compte is not None
        assert compte.technique is True
        assert len(mdp) > 20  # généré, pas choisi
        assert compte.teams == ["DATA-01"]

    async def test_description_obligatoire(self, db):
        """Un compte de service sans raison d'être documentée devient
        impossible à auditer six mois plus tard."""
        compte, message = await user.creer_compte_technique("x", "x@interne", "   ")
        assert compte is None
        assert "description" in message.lower()

    async def test_connexion_interactive_refusee(self, db, equipe_data):
        from Engine.models import authenticate

        _, mdp = await user.creer_compte_technique(
            "agent-rag", "agent-rag@interne", "Agent RAG", "DATA-01"
        )
        assert await authenticate("agent-rag@interne", mdp) is None
        assert await authenticate("agent-rag", mdp) is None

    async def test_compte_humain_non_technique(self, karim):
        recharge = await user.load("karim@entreprise.fr")
        assert recharge.technique is False

    async def test_inventaire(self, catalogue, db, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        await user.creer_compte_technique(
            "agent-rag", "agent-rag@interne", "Agent RAG", "DATA-01"
        )
        inventaire = await user.comptes_techniques()

        assert [c["login"] for c in inventaire] == ["agent-rag"]
        assert inventaire[0]["equipes"] == ["DATA-01"]
        assert inventaire[0]["modeles"] == ["llama3.1:8b"]
        # Le compte humain n'apparaît pas dans l'inventaire technique
        assert "ksoumahoro" not in [c["login"] for c in inventaire]

    async def test_habilitations_comme_un_compte_humain(self, catalogue, equipe_data):
        await equipe_data.add_model("llama3.1:8b")
        compte, _ = await user.creer_compte_technique(
            "agent-rag", "agent-rag@interne", "Agent RAG", "DATA-01"
        )
        assert await compte.can_use_model("llama3.1:8b")
        assert not await compte.can_use_model("mistral:7b")
