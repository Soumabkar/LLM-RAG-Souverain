"""Tests de la classe `user` contre une vraie base PostgreSQL."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.models import user
from app.security import verify_password

pytestmark = pytest.mark.db


class TestCreateUser:
    def test_creation_nominale(self, equipe_data):
        u = user("jdoe", "Secret!2026", "jdoe@entreprise.fr")
        res = u.create_user("DATA-01")
        assert res.ok
        assert u.teams == ["DATA-01"]
        assert user.find("jdoe@entreprise.fr") is not None

    def test_creation_sans_equipe(self, db):
        u = user("solo", "Secret!2026", "solo@entreprise.fr")
        assert u.create_user()
        assert u.teams == []

    def test_email_deja_pris(self, karim):
        doublon = user("autre_login", "Secret!2026", "karim@entreprise.fr")
        res = doublon.create_user("DATA-01")
        assert not res.ok
        assert "existe déjà" in res.message

    def test_login_deja_pris(self, karim):
        doublon = user("ksoumahoro", "Secret!2026", "autre@entreprise.fr")
        res = doublon.create_user("DATA-01")
        assert not res.ok
        assert "login" in res.message.lower()
        assert user.find("autre@entreprise.fr") is None, "le compte ne doit pas exister"

    def test_equipe_inexistante(self, db):
        u = user("ghost", "Secret!2026", "ghost@entreprise.fr")
        res = u.create_user("TEAM-INCONNUE")
        assert not res.ok
        assert "n'existe pas" in res.message

    def test_atomicite_si_equipe_inexistante(self, db):
        """L'utilisateur ne doit pas rester en base sans son adhésion."""
        user("ghost", "Secret!2026", "ghost@entreprise.fr").create_user("TEAM-INCONNUE")
        assert user.find("ghost@entreprise.fr") is None

    def test_mot_de_passe_stocke_hache(self, karim, sql):
        stocke = sql(
            "SELECT password FROM llm_souverain.user_llm WHERE email = %s",
            ("karim@entreprise.fr",),
        )[0]["password"]
        assert stocke != "MotDePasse!2026"
        assert verify_password("MotDePasse!2026", stocke)

    def test_mot_de_passe_long_accepte(self, equipe_data):
        """Régression : bcrypt refuse au-delà de 72 octets, ce qui faisait
        remonter une ValueError non capturée au lieu d'un Result."""
        u = user("longpw", "é" * 60, "long@entreprise.fr")
        assert u.create_user("DATA-01")
        assert u.check_password("é" * 60)


class TestValidation:
    def test_login_trop_long(self, equipe_data):
        res = user("L" * 60, "Secret!2026", "x@entreprise.fr").create_user("DATA-01")
        assert not res.ok
        assert "login" in res.message.lower()
        assert "60" in res.message, "le message doit être exploitable par l'IHM"

    def test_email_trop_long(self, equipe_data):
        res = user("x", "Secret!2026", "a" * 300 + "@x.fr").create_user("DATA-01")
        assert not res.ok
        assert "email" in res.message.lower()

    @pytest.mark.parametrize("champ", ["login", "email", "password"])
    def test_champ_obligatoire_vide(self, equipe_data, champ):
        valeurs = {"login": "x", "password": "p", "email": "x@x.fr"}
        valeurs[champ] = "   "
        res = user(**valeurs).create_user("DATA-01")
        assert not res.ok
        assert "obligatoire" in res.message


class TestDeleteUser:
    def test_suppression_nominale(self, karim):
        assert karim.delete_user()
        assert user.find("karim@entreprise.fr") is None

    def test_suppression_utilisateur_absent(self, db):
        res = user("x", "p", "absent@entreprise.fr").delete_user()
        assert not res.ok
        assert "n'existe pas" in res.message

    def test_les_adhesions_partent_en_cascade(self, karim, sql):
        karim.delete_user()
        assert sql("SELECT * FROM llm_souverain.team_member") == []

    def test_l_equipe_survit(self, karim, sql):
        karim.delete_user()
        assert len(sql("SELECT * FROM llm_souverain.team_llm")) == 1


class TestUpdateUser:
    def test_mise_a_jour_du_login(self, karim):
        karim.login = "k.soumahoro"
        assert karim.update_user()
        assert user.find("karim@entreprise.fr")["login"] == "k.soumahoro"

    def test_utilisateur_absent(self, db):
        res = user("x", "p", "absent@entreprise.fr").update_user()
        assert not res.ok

    def test_login_en_conflit(self, karim, equipe_data):
        autre = user("autre", "Secret!2026", "autre@entreprise.fr")
        autre.create_user("DATA-01")
        autre.login = "ksoumahoro"
        res = autre.update_user()
        assert not res.ok
        assert "login" in res.message.lower()

    def test_rechargement_puis_sauvegarde_ne_casse_pas_le_mot_de_passe(self, karim):
        """`load()` met le hash dans self.password ; `update_user()` ne doit
        pas le hacher une seconde fois, sinon le mot de passe est perdu."""
        recharge = user.load("karim@entreprise.fr")
        recharge.login = "nouveau_login"
        assert recharge.update_user()
        assert recharge.check_password("MotDePasse!2026")

    def test_changement_de_mot_de_passe(self, karim):
        karim.set_password("NouveauMdp!2026")
        assert karim.update_user()
        assert karim.check_password("NouveauMdp!2026")
        assert not karim.check_password("MotDePasse!2026")


class TestChangeEmail:
    def test_changement_nominal(self, karim):
        assert karim.change_email("karim.soumahoro@entreprise.fr")
        assert karim.email == "karim.soumahoro@entreprise.fr"
        assert user.find("karim@entreprise.fr") is None

    def test_les_adhesions_suivent(self, karim, sql):
        karim.change_email("nouveau@entreprise.fr")
        adhesions = sql("SELECT email FROM llm_souverain.team_member")
        assert adhesions[0]["email"] == "nouveau@entreprise.fr"

    def test_email_deja_pris(self, karim, equipe_data):
        autre = user("autre", "Secret!2026", "autre@entreprise.fr")
        autre.create_user("DATA-01")
        res = autre.change_email("karim@entreprise.fr")
        assert not res.ok
        assert "déjà utilisé" in res.message


class TestLoadEtFind:
    def test_load_restitue_les_equipes(self, karim):
        recharge = user.load("karim@entreprise.fr")
        assert recharge.login == "ksoumahoro"
        assert recharge.teams == ["DATA-01"]

    def test_load_utilisateur_absent(self, db):
        assert user.load("absent@entreprise.fr") is None

    def test_propriete_code_team_compatibilite(self, karim):
        assert karim.code_team == "DATA-01"

    def test_code_team_none_si_aucune_equipe(self, db):
        u = user("solo", "Secret!2026", "solo@entreprise.fr")
        u.create_user()
        assert u.code_team is None


class TestCheckPassword:
    def test_mot_de_passe_correct(self, karim):
        assert karim.check_password("MotDePasse!2026")

    def test_mot_de_passe_incorrect(self, karim):
        assert not karim.check_password("mauvais")

    def test_utilisateur_absent(self, db):
        assert not user("x", "p", "absent@entreprise.fr").check_password("p")


class TestConcurrence:
    def test_un_seul_gagnant_sur_creation_simultanee(self, equipe_data):
        """Dix créations parallèles du même email : une seule doit réussir.
        C'est ce que garantit ON CONFLICT, là où un SELECT puis INSERT
        laisserait passer des doublons."""

        def creer(i: int) -> bool:
            return user(f"login{i}", "Secret!2026", "race@entreprise.fr").create_user(
                "DATA-01"
            ).ok

        with ThreadPoolExecutor(max_workers=10) as pool:
            resultats = list(pool.map(creer, range(10)))

        assert sum(resultats) == 1
