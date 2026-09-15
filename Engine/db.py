"""Accès asynchrone à la base PostgreSQL du schéma llm_souverain.

Le pool est asynchrone (`psycopg_pool.AsyncConnectionPool`) : les requêtes
rendent la main à la boucle d'événements pendant l'attente réseau, au lieu
de la bloquer. C'est ce qui permet à Chainlit de continuer à servir les
autres sessions pendant qu'une requête est en cours.

Toutes les fonctions publiques sont des coroutines ou des gestionnaires de
contexte asynchrones : elles s'utilisent avec `await` ou `async with`.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

try:  # chargement facultatif du .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------
# Boucle d'événements compatible (Windows)
#
# Depuis Python 3.8, Windows utilise ProactorEventLoop par défaut. psycopg
# en mode asynchrone ne sait pas fonctionner dessus et refuse toute
# connexion : « Psycopg cannot use the 'ProactorEventLoop' to run in async
# mode ». On bascule sur SelectorEventLoop dès l'import, ce qui couvre
# aussi bien la démo que Chainlit et pytest.
#
# Contrepartie : SelectorEventLoop plafonne à 512 sockets et ne gère pas
# les sous-processus asyncio. Sans objet ici.
# ---------------------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


SCHEMA = "llm_souverain"

_pool: AsyncConnectionPool | None = None
# Protège la création du pool : sans lui, deux coroutines démarrant en même
# temps en créeraient chacune un, et l'un des deux fuirait.
_lock = asyncio.Lock()


def _env(name: str, default: str = "") -> str:
    """Lit une variable d'environnement en la nettoyant.

    Le .strip() n'est pas cosmétique : un fichier .env enregistré sous
    Windows (CRLF) laisse un \r en fin de valeur, qui produit un
    « failed to resolve host 'localhost\r' » invisible à la lecture.
    """
    valeur = os.getenv(name)
    return (valeur if valeur is not None else default).strip()


def _conninfo() -> str:
    return psycopg.conninfo.make_conninfo(
        host=_env("POSTGRES_HOST"),
        port=_env("POSTGRES_PORT"),
        dbname=_env("POSTGRES_DB"),
        user=_env("POSTGRES_USER"),
        password=_env("POSTGRES_PASSWORD"),
    )


async def get_pool() -> AsyncConnectionPool:
    """Pool de connexions, créé une seule fois par boucle d'événements."""
    global _pool
    if _pool is None:
        async with _lock:
            if _pool is None:  # revérifié : un autre appel a pu passer
                pool = AsyncConnectionPool(
                    conninfo=_conninfo(),
                    min_size=1,
                    max_size=int(_env("POSTGRES_POOL_MAX", "10")),
                    kwargs={
                        "row_factory": dict_row,
                        "options": f"-c search_path={SCHEMA},public",
                    },
                    # Obligatoire en asynchrone : le pool ne peut pas
                    # s'ouvrir dans le constructeur, il n'y a pas encore
                    # de boucle d'événements à ce moment-là.
                    open=False,
                )
                try:
                    await pool.open(wait=True, timeout=10)
                except Exception:
                    # Sans ce nettoyage, un pool mort resterait en cache et
                    # masquerait l'erreur réelle derrière des PoolClosed.
                    await pool.close()
                    raise
                _pool = pool
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncIterator[psycopg.AsyncConnection]:
    """Connexion empruntée au pool.

    Le bloc `async with` de psycopg valide la transaction à la sortie
    normale et fait un rollback si une exception remonte.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn


async def close_pool() -> None:
    """À appeler à l'arrêt de l'application."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
