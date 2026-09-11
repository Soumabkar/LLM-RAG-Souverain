"""Tests de la classe `team` et de l'appartenance N-N."""

from __future__ import annotations

import pytest

from app.models import team, user

pytestmark = pytest.mark.db


class TestCreateTeam:
    def test_creation_nominale(self, db):
        res = team("DATA-01", "Data Platform", "data@entreprise.fr").create_team()
        assert res.ok

    def test_creation_sans_email(self, db):
        assert team("DATA-01", "Data Platform").create_team()

    def test_code_deja_pris(self, equipe_data):
        res = team("DATA-01", "Autre nom").create_team()
        assert not res.ok
        assert "existe déjà" in res.message

    def test_nom_trop_long(self, db):
        res = team("T1", "N" * 60).create_team()
        assert not res.ok
        assert "60" in res.message

    def test_nom_obligatoire(self, db):
        res = team("T1", "").create_team()
        assert not res.ok
        assert "obligatoire" in res.message


class TestUpdateTeam:
    def test_mise_a_jour_du_nom(self, equipe_data):
        equipe_data.team_name = "Data & IA"
        assert equipe_data.update_team()
        assert team.load("DATA-01").team_name == "Data & IA"

    def test_equipe_absente(self, db):
        res = team("INCONNUE", "X").update_team()
        assert not res.ok
        assert "n'existe pas" in res.message

    def test_email_non_ecrase_si_non_fourni(self, equipe_data):
        """Régression : `team("DATA-01", "Nouveau nom").update_team()`
        effaçait silencieusement l'email de contact."""
        team("DATA-01", "Nouveau nom").update_team()
        assert team.load("DATA-01").email == "data@entreprise.fr"

    def test_email_videable_avec_chaine_vide(self, equipe_data):
        equipe_data.email = ""
        equipe_data.update_team()
        assert team.load("DATA-01").email == ""


class TestDeleteTeam:
    def test_suppression_nominale(self, equipe_data):
        assert equipe_data.delete_team()
        assert team.load("DATA-01") is None

    def test_equipe_absente(self, db):
        res = team("INCONNUE", "X").delete_team()
        assert not res.ok

    def test_les_comptes_survivent(self, karim):
        team.load("DATA-01").delete_team()
        assert user.find("karim@entreprise.fr") is not None

    def test_adhesions_supprimees_en_cascade(self, karim, sql):
        team.load("DATA-01").delete_team()
        assert sql("SELECT * FROM llm_souverain.team_member") == []
        assert karim.load_teams() == []

    def test_message_indique_le_nombre_de_membres(self, karim):
        res = team.load("DATA-01").delete_team()
        assert "1 adhésion" in res.message


class TestAppartenance:
    def test_ajout_nominal(self, karim, equipe_secu):
        res = equipe_secu.add_member_team(karim)
        assert res.ok
        assert equipe_secu.members == ["karim@entreprise.fr"]
        assert karim.teams == ["DATA-01", "SEC-01"]

    def test_multi_appartenance(self, karim, equipe_secu):
        equipe_secu.add_member_team(karim)
        assert set(karim.load_teams()) == {"DATA-01", "SEC-01"}

    def test_ajout_deux_fois(self, karim, equipe_secu):
        equipe_secu.add_member_team(karim)
        res = equipe_secu.add_member_team(karim)
        assert not res.ok
        assert "déjà membre" in res.message

    def test_equipe_inexistante(self, karim):
        res = team("INCONNUE", "X").add_member_team(karim)
        assert not res.ok
        assert "équipe" in res.message.lower()

    def test_utilisateur_inexistant(self, equipe_data):
        fantome = user("ghost", "p", "ghost@entreprise.fr")
        res = equipe_data.add_member_team(fantome)
        assert not res.ok
        assert "utilisateur" in res.message.lower()

    def test_retrait_nominal(self, karim, equipe_secu):
        equipe_secu.add_member_team(karim)
        assert equipe_secu.delete_member_team(karim)
        assert karim.teams == ["DATA-01"]

    def test_retrait_ne_supprime_pas_le_compte(self, karim, equipe_data):
        equipe_data.delete_member_team(karim)
        assert user.find("karim@entreprise.fr") is not None

    def test_retrait_d_un_non_membre(self, karim, equipe_secu):
        res = equipe_secu.delete_member_team(karim)
        assert not res.ok
        assert "n'est pas membre" in res.message

    def test_join_team_et_leave_team(self, karim, equipe_secu):
        assert karim.join_team("SEC-01")
        assert karim.is_member_of("SEC-01")
        assert karim.leave_team("SEC-01")
        assert not karim.is_member_of("SEC-01")

    def test_load_members_trie(self, equipe_data):
        for prenom in ("zoe", "alice", "marc"):
            user(prenom, "Secret!2026", f"{prenom}@entreprise.fr").create_user("DATA-01")
        assert equipe_data.load_members() == [
            "alice@entreprise.fr",
            "marc@entreprise.fr",
            "zoe@entreprise.fr",
        ]

    def test_load_members_equipe_vide(self, equipe_data):
        assert equipe_data.load_members() == []


class TestVueUserTeams:
    def test_agregation_des_equipes(self, karim, equipe_secu, sql):
        equipe_secu.add_member_team(karim)
        ligne = sql(
            "SELECT teams FROM llm_souverain.v_user_teams WHERE email = %s",
            ("karim@entreprise.fr",),
        )[0]
        assert ligne["teams"] == ["DATA-01", "SEC-01"]

    def test_utilisateur_sans_equipe(self, db, sql):
        user("solo", "Secret!2026", "solo@entreprise.fr").create_user()
        ligne = sql(
            "SELECT teams FROM llm_souverain.v_user_teams WHERE email = %s",
            ("solo@entreprise.fr",),
        )[0]
        assert ligne["teams"] == [], "tableau vide attendu, pas NULL"
