"""Fixtures partagées.

Les tests marqués `db` exigent un PostgreSQL joignable avec le schéma
llm_souverain déjà créé (`docker compose up -d`). S'il est absent, ils
sont ignorés plutôt que mis en échec : `pytest -m "not db"` suffit alors
à jouer les tests purement unitaires.

La couche d'accès étant asynchrone, les fixtures qui touchent la base le
sont aussi. `asyncio_mode = auto` (pytest.ini) évite d'avoir à décorer
chaque test.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from Engine.db import close_pool, get_connection
from Engine.models import model_llm, team, user

TABLES = (
    "llm_souverain.team_model",
    "llm_souverain.model_llm",
    "llm_souverain.team_member",
    "llm_souverain.user_llm",
    "llm_souverain.team_llm",
)


async def _database_available() -> bool:
    try:
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT to_regclass('llm_souverain.team_member') AS t")
            row = await cur.fetchone()
            return row["t"] is not None
    except Exception:  # noqa: BLE001 - toute erreur signifie « pas de base »
        return False


async def _truncate() -> None:
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(f"TRUNCATE {', '.join(TABLES)} CASCADE")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def database():
    if not await _database_available():
        pytest.skip(
            "PostgreSQL injoignable ou schéma llm_souverain absent "
            "(lancer `docker compose up -d`)",
            allow_module_level=True,
        )
    yield
    await close_pool()


@pytest_asyncio.fixture(loop_scope="session")
async def db(database):
    """Base vide avant chaque test."""
    await _truncate()
    yield
    await _truncate()


@pytest_asyncio.fixture(loop_scope="session")
async def sql(database):
    """Exécute une requête brute et renvoie les lignes."""

    async def _run(query: str, params: tuple = ()):
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall() if cur.description else None

    return _run


@pytest_asyncio.fixture(loop_scope="session")
async def equipe_data(db) -> team:
    t = team("DATA-01", "Data Platform", "data@entreprise.fr")
    assert await t.create_team()
    return t


@pytest_asyncio.fixture(loop_scope="session")
async def equipe_secu(db) -> team:
    t = team("SEC-01", "Cybersécurité", "secu@entreprise.fr")
    assert await t.create_team()
    return t


@pytest_asyncio.fixture(loop_scope="session")
async def karim(equipe_data) -> user:
    u = user("ksoumahoro", "MotDePasse!2026", "karim@entreprise.fr")
    assert await u.create_user("DATA-01")
    return u


@pytest_asyncio.fixture(loop_scope="session")
async def catalogue(db) -> list[str]:
    """Trois modèles au catalogue, aucun encore habilité."""
    await model_llm("llama3.1:8b", "Llama 3.1 8B", "Généraliste").create_model(verify=False)
    await model_llm("mistral:7b", "Mistral 7B", "Bon en français").create_model(verify=False)
    await model_llm("qwen2.5-coder:7b", "Qwen Coder", "Code").create_model(verify=False)
    return ["llama3.1:8b", "mistral:7b", "qwen2.5-coder:7b"]