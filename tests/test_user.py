"""Tests de la classe `user` contre une vraie base PostgreSQL."""

from __future__ import annotations

import asyncio

import pytest

from Engine.models import user
from Engine.security import verify_password

pytestmark = pytest.mark.db


class TestCreateUser:
    async def test_creation_nominale(self, equipe_data):
        u = user("jdoe", "Secret!2026", "jdoe@entreprise.fr")
        res = await u.create_user("DATA-01")
        assert res.ok
        assert u.teams == ["DATA-01"]
        assert await user.find("jdoe@entreprise.fr") is not None

    async def test_creation_sans_equipe(self, db):
        u = user("solo", "Secret!2026", "solo@entreprise.fr")
        assert await u.create_user()
        assert u.teams == []

    async def test_email_deja_pris(self, karim):
        doublon = user("autre_login", "Secret!2026", "karim@entreprise.fr")
        res = await doublon.create_user("DATA-01")
        assert not res.ok
        assert "existe déjà" in res.message

    async def test_login_deja_pris(self, karim):
        doublon = user("ksoumahoro", "Secret!2026", "autre@entreprise.fr")
        res = await doublon.create_user("DATA-01")
        assert not res.ok
        assert "login" in res.message.lower()
        assert await user.find("autre@entreprise.fr") is None, "le compte ne doit pas exister"

    async def test_equipe_inexistante(self, db):
        u = user("ghost", "Secret!2026", "ghost@entreprise.fr")
        res = await u.create_user("TEAM-INCONNUE")
        assert not res.ok
        assert "n'existe pas" in res.message

    async def test_atomicite_si_equipe_inexistante(self, db):
        """L'utilisateur ne doit pas rester en base sans son adhésion."""
        await user("ghost", "Secret!2026", "ghost@entreprise.fr").create_user("TEAM-INCONNUE")
        assert await user.find("ghost@entreprise.fr") is None

    async def test_mot_de_passe_stocke_hache(self, karim, sql):
        stocke = (await sql(
            "SELECT password FROM llm_souverain.user_llm WHERE email = %s",
            ("karim@entreprise.fr",),
        ))[0]["password"]
        assert stocke != "MotDePasse!2026"
        assert verify_password("MotDePasse!2026", stocke)

    async def test_mot_de_passe_long_accepte(self, equipe_data):
        """Régression : bcrypt refuse au-delà de 72 octets, ce qui faisait
        remonter une ValueError non capturée au lieu d'un Result."""
        u = user("longpw", "é" * 60, "long@entreprise.fr")
        assert await u.create_user("DATA-01")
        assert await u.check_password("é" * 60)


class TestValidation:
    async def test_login_trop_long(self, equipe_data):
        res = await user("L" * 60, "Secret!2026", "x@entreprise.fr").create_user("DATA-01")
        assert not res.ok
        assert "login" in res.message.lower()
        assert "60" in res.message, "le message doit être exploitable par l'IHM"

    async def test_email_trop_long(self, equipe_data):
        res = await user("x", "Secret!2026", "a" * 300 + "@x.fr").create_user("DATA-01")
        assert not res.ok
        assert "email" in res.message.lower()

    @pytest.mark.parametrize("champ", ["login", "email", "password"])
    async def test_champ_obligatoire_vide(self, equipe_data, champ):
        valeurs = {"login": "x", "password": "p", "email": "x@x.fr"}
        valeurs[champ] = "   "
        res = await user(**valeurs).create_user("DATA-01")
        assert not res.ok
        assert "obligatoire" in res.message


class TestDeleteUser:
    async def test_suppression_nominale(self, karim):
        assert await karim.delete_user()
        assert await user.find("karim@entreprise.fr") is None

    async def test_suppression_utilisateur_absent(self, db):
        res = await user("x", "p", "absent@entreprise.fr").delete_user()
        assert not res.ok
        assert "n'existe pas" in res.message

    async def test_les_adhesions_partent_en_cascade(self, karim, sql):
        await karim.delete_user()
        assert await sql("SELECT * FROM llm_souverain.team_member") == []

    async def test_l_equipe_survit(self, karim, sql):
        await karim.delete_user()
        assert len(await sql("SELECT * FROM llm_souverain.team_llm")) == 1


class TestUpdateUser:
    async def test_mise_a_jour_du_login(self, karim):
        karim.login = "k.soumahoro"
        assert await karim.update_user()
        assert (await user.find("karim@entreprise.fr"))["login"] == "k.soumahoro"

    async def test_utilisateur_absent(self, db):
        res = await user("x", "p", "absent@entreprise.fr").update_user()
        assert not res.ok

    async def test_login_en_conflit(self, karim, equipe_data):
        autre = user("autre", "Secret!2026", "autre@entreprise.fr")
        await autre.create_user("DATA-01")
        autre.login = "ksoumahoro"
        res = await autre.update_user()
        assert not res.ok
        assert "login" in res.message.lower()

    async def test_rechargement_puis_sauvegarde_ne_casse_pas_le_mot_de_passe(self, karim):
        """`load()` met le hash dans self.password ; `update_user()` ne doit
        pas le hacher une seconde fois, sinon le mot de passe est perdu."""
        recharge = await user.load("karim@entreprise.fr")
        recharge.login = "nouveau_login"
        assert await recharge.update_user()
        assert await recharge.check_password("MotDePasse!2026")

    async def test_changement_de_mot_de_passe(self, karim):
        karim.set_password("NouveauMdp!2026")
        assert await karim.update_user()
        assert await karim.check_password("NouveauMdp!2026")
        assert not await karim.check_password("MotDePasse!2026")


class TestChangeEmail:
    async def test_changement_nominal(self, karim):
        assert await karim.change_email("karim.soumahoro@entreprise.fr")
        assert karim.email == "karim.soumahoro@entreprise.fr"
        assert await user.find("karim@entreprise.fr") is None

    async def test_les_adhesions_suivent(self, karim, sql):
        await karim.change_email("nouveau@entreprise.fr")
        adhesions = await sql("SELECT email FROM llm_souverain.team_member")
        assert adhesions[0]["email"] == "nouveau@entreprise.fr"

    async def test_email_deja_pris(self, karim, equipe_data):
        autre = user("autre", "Secret!2026", "autre@entreprise.fr")
        await autre.create_user("DATA-01")
        res = await autre.change_email("karim@entreprise.fr")
        assert not res.ok
        assert "déjà utilisé" in res.message


class TestLoadEtFind:
    async def test_load_restitue_les_equipes(self, karim):
        recharge = await user.load("karim@entreprise.fr")
        assert recharge.login == "ksoumahoro"
        assert recharge.teams == ["DATA-01"]

    async def test_load_utilisateur_absent(self, db):
        assert await user.load("absent@entreprise.fr") is None

    async def test_propriete_code_team_compatibilite(self, karim):
        assert karim.code_team == "DATA-01"

    async def test_code_team_none_si_aucune_equipe(self, db):
        u = user("solo", "Secret!2026", "solo@entreprise.fr")
        await u.create_user()
        assert u.code_team is None


class TestCheckPassword:
    async def test_mot_de_passe_correct(self, karim):
        assert await karim.check_password("MotDePasse!2026")

    async def test_mot_de_passe_incorrect(self, karim):
        assert not await karim.check_password("mauvais")

    async def test_utilisateur_absent(self, db):
        assert not await user("x", "p", "absent@entreprise.fr").check_password("p")


class TestConcurrence:
    async def test_un_seul_gagnant_sur_creation_simultanee(self, equipe_data):
        """Dix créations parallèles du même email : une seule doit réussir.
        C'est ce que garantit ON CONFLICT, là où un SELECT puis INSERT
        laisserait passer des doublons."""

        async def creer(i: int) -> bool:
            res = await user(
                f"login{i}", "Secret!2026", "race@entreprise.fr"
            ).create_user("DATA-01")
            return res.ok

        resultats = await asyncio.gather(*(creer(i) for i in range(10)))

        assert sum(resultats) == 1
