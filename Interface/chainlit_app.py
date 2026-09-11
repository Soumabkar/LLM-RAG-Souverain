"""Intégration Chainlit : chainlit run app/chainlit_app.py -w

Une instance d'`ai_model` par session, pour que l'historique ne fuite pas
d'un utilisateur à l'autre.

L'authentification par mot de passe s'appuie sur llm_souverain.user_llm.
Elle exige CHAINLIT_AUTH_SECRET dans l'environnement, sans quoi Chainlit
ignore purement et simplement le callback et n'affiche aucun écran de
connexion.
"""

from __future__ import annotations

import logging
import os

import chainlit as cl

from Model.ai_model import AIModelError, ai_model
from Engine.db import close_pool
from LLM.llm import LLMAccessError, llm
from Engine.models import authenticate, user

logging.basicConfig(level=logging.INFO)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
DEFAULT_MODEL = os.getenv("LLM_MODEL")


# =====================================================================
# Authentification
# =====================================================================
@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    """Authentifie contre llm_souverain.user_llm.

    `username` accepte l'email ou le login. Renvoyer None fait afficher à
    Chainlit un message d'échec générique : ne pas préciser si c'est le
    compte ou le mot de passe qui est en cause.

    Nécessite CHAINLIT_AUTH_SECRET dans l'environnement
    (`chainlit create-secret` pour en générer un).
    """
    compte = await authenticate(username, password)
    if compte is None:
        return None

    # L'identifiant devient l'email : c'est la clé primaire, donc ce que
    # `user.load()` attend à l'ouverture de la session.
    return cl.User(
        identifier=compte.email,
        display_name=compte.login,
        metadata={"provider": "credentials", "teams": compte.teams},
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    model = ai_model.from_ollama(model=DEFAULT_MODEL, base_url=OLLAMA_URL)

    if not await model.health_check():
        await cl.Message(
            content=f"⚠️ Le modèle `{DEFAULT_MODEL}` n'est pas disponible sur {OLLAMA_URL}."
        ).send()
        return

    # Identité posée par `auth_callback`. En développement sans
    # CHAINLIT_AUTH_SECRET, repli sur DEV_USER_EMAIL.
    cl_user = cl.user_session.get("user")
    email = cl_user.identifier if cl_user else os.getenv("DEV_USER_EMAIL", "")

    compte = await user.load(email) if email else None
    if compte is None:
        await cl.Message(content=f"Compte inconnu : `{email}`.").send()
        return

    assistant = llm(model, compte)
    try:
        await assistant.authorize()
    except LLMAccessError as exc:
        await cl.Message(content=f"Accès refusé : {exc}").send()
        return

    cl.user_session.set("assistant", assistant)
    equipes = ", ".join(compte.teams) or "aucune équipe"
    await cl.Message(
        content=f"Bonjour {compte.login} ({equipes}) — modèle `{DEFAULT_MODEL}`."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    assistant: llm | None = cl.user_session.get("assistant")
    if assistant is None:
        await cl.Message(content="Session non initialisée, rechargez la page.").send()
        return

    reply = cl.Message(content="")
    await reply.send()

    try:
        async for token in assistant.stream_response_llm(message.content):
            await reply.stream_token(token)
    except LLMAccessError as exc:
        reply.content = f"🔒 {exc}"
    except AIModelError as exc:
        reply.content = f"❌ {exc}"

    await reply.update()


@cl.on_chat_end
async def on_chat_end() -> None:
    assistant: llm | None = cl.user_session.get("assistant")
    if assistant is not None:
        await assistant.model.close()


@cl.on_stop
async def on_stop() -> None:
    """Le pool de connexions est partagé par toutes les sessions :
    il ne se ferme qu'à l'arrêt du serveur, pas à chaque déconnexion."""
    await close_pool()
