"""Tests des garanties portées par le schéma lui-même.

Ces contraintes protègent la base même si un autre client (psql, un job
Spark, un futur service) écrit sans passer par les classes Python.
"""

from __future__ import annotations

import psycopg
import pytest

from Engine.db import get_connection

pytestmark = pytest.mark.db


async def _executer(requete: str, params: tuple = ()):
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(requete, params)


class TestStructure:
    @pytest.mark.parametrize("table", ["team_llm", "user_llm", "team_member"])
    async def test_table_presente(self, sql, table):
        assert (await sql("SELECT to_regclass(%s) AS t", (f"llm_souverain.{table}",)))[0]["t"]

    async def test_colonne_code_team_supprimee_de_user_llm(self, sql):
        """Source de vérité unique : l'appartenance ne vit que dans
        team_member."""
        colonnes = await sql(
            """
            SELECT column_name FROM information_schema.columns
             WHERE table_schema = 'llm_souverain' AND table_name = 'user_llm'
            """
        )
        assert "code_team" not in {c["column_name"] for c in colonnes}

    async def test_vue_presente(self, sql):
        assert (await sql("SELECT to_regclass('llm_souverain.v_user_teams') AS t"))[0]["t"]


class TestContraintes:
    async def test_email_unique(self, db):
        await _executer(
            "INSERT INTO llm_souverain.user_llm VALUES ('a', 'h', 'x@x.fr')"
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            await _executer("INSERT INTO llm_souverain.user_llm VALUES ('b', 'h', 'x@x.fr')")

    async def test_login_unique(self, db):
        await _executer("INSERT INTO llm_souverain.user_llm VALUES ('a', 'h', 'x@x.fr')")
        with pytest.raises(psycopg.errors.UniqueViolation):
            await _executer("INSERT INTO llm_souverain.user_llm VALUES ('a', 'h', 'y@x.fr')")

    async def test_adhesion_unique(self, karim):
        with pytest.raises(psycopg.errors.UniqueViolation):
            await _executer(
                "INSERT INTO llm_souverain.team_member (code_team, email) VALUES (%s, %s)",
                ("DATA-01", "karim@entreprise.fr"),
            )

    async def test_adhesion_vers_equipe_inexistante_refusee(self, karim):
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await _executer(
                "INSERT INTO llm_souverain.team_member (code_team, email) VALUES (%s, %s)",
                ("INCONNUE", "karim@entreprise.fr"),
            )

    async def test_adhesion_vers_utilisateur_inexistant_refusee(self, equipe_data):
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await _executer(
                "INSERT INTO llm_souverain.team_member (code_team, email) VALUES (%s, %s)",
                ("DATA-01", "absent@x.fr"),
            )

    async def test_login_limite_a_50_caracteres(self, db):
        with pytest.raises(psycopg.errors.StringDataRightTruncation):
            await _executer(
                "INSERT INTO llm_souverain.user_llm VALUES (%s, 'h', 'x@x.fr')",
                ("L" * 51,),
            )

    async def test_hash_de_256_caracteres_accepte(self, db):
        """La colonne doit tenir le format bcrypt-sha256 (74 caractères)
        avec de la marge pour un futur argon2."""
        await _executer(
            "INSERT INTO llm_souverain.user_llm VALUES ('a', %s, 'x@x.fr')",
            ("h" * 256,),
        )


class TestCascades:
    async def test_suppression_equipe_cascade_sur_adhesions(self, karim, sql):
        await _executer("DELETE FROM llm_souverain.team_llm WHERE code_team = 'DATA-01'")
        assert await sql("SELECT * FROM llm_souverain.team_member") == []
        assert len(await sql("SELECT * FROM llm_souverain.user_llm")) == 1

    async def test_suppression_utilisateur_cascade_sur_adhesions(self, karim, sql):
        await _executer("DELETE FROM llm_souverain.user_llm WHERE email = 'karim@entreprise.fr'")
        assert await sql("SELECT * FROM llm_souverain.team_member") == []
        assert len(await sql("SELECT * FROM llm_souverain.team_llm")) == 1

    async def test_renommage_equipe_cascade(self, karim, sql):
        await _executer(
            "UPDATE llm_souverain.team_llm SET code_team = 'DATA-02' WHERE code_team = 'DATA-01'"
        )
        assert (await sql("SELECT code_team FROM llm_souverain.team_member"))[0]["code_team"] == "DATA-02"

    async def test_changement_email_cascade(self, karim, sql):
        await _executer(
            "UPDATE llm_souverain.user_llm SET email = 'neo@x.fr' WHERE email = 'karim@entreprise.fr'"
        )
        assert (await sql("SELECT email FROM llm_souverain.team_member"))[0]["email"] == "neo@x.fr"


class TestSearchPath:
    async def test_tables_accessibles_sans_prefixe(self, sql):
        """Le pool positionne search_path : une requête sans préfixe de
        schéma doit fonctionner."""
        assert (await sql("SELECT count(*) AS n FROM team_member"))[0]["n"] >= 0

    async def test_joined_at_rempli_automatiquement(self, karim, sql):
        assert (await sql("SELECT joined_at FROM llm_souverain.team_member"))[0]["joined_at"]
