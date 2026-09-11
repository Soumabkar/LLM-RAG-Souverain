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