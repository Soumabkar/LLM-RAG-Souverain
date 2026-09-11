"""Intégration Chainlit : chainlit run Interface/chainlit_app.py -w

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
from chainlit.input_widget import Select

from Model.ai_model import AIModelError, ai_model
from Engine.db import close_pool
from LLM.llm import LLMAccessError, llm
from Engine.models import authenticate, user

logging.basicConfig(level=logging.INFO)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")


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

    # Les modèles viennent des équipes de l'utilisateur, pas d'une constante.
    modeles = await compte.allowed_models()
    if not modeles:
        await cl.Message(
            content=(
                f"Aucun modèle n'est autorisé pour `{compte.email}`. "
                "Demande à un administrateur de rattacher ton équipe à un modèle."
            )
        ).send()
        return

    # Modèle d'ouverture : celui du .env s'il est autorisé, sinon le premier.
    codes = [m["code_model"] for m in modeles]
    model.model = DEFAULT_MODEL if DEFAULT_MODEL in codes else codes[0]

    assistant = llm(model, compte)
    try:
        await assistant.authorize()
    except LLMAccessError as exc:
        await cl.Message(content=f"Accès refusé : {exc}").send()
        return

    cl.user_session.set("assistant", assistant)

    # Sélecteur de modèle, limité aux habilitations. ChatSettings plutôt que
    # les chat profiles : changer de profil repart d'une conversation vide,
    # alors qu'ici l'historique survit à la bascule.
    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Modèle",
                # `items` porte à la fois le libellé et la valeur ; le passer
                # en même temps que `values` est refusé par Chainlit.
                items={m["display_name"]: m["code_model"] for m in modeles},
                initial_value=model.model,
            )
        ]
    ).send()

    equipes = ", ".join(compte.teams) or "aucune équipe"
    courant = next(m["display_name"] for m in modeles if m["code_model"] == model.model)
    await cl.Message(
        content=(
            f"Bonjour {compte.login} ({equipes}).\n"
            f"Modèle : **{courant}** — {len(modeles)} modèle(s) disponible(s) "
            "dans les paramètres."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """Bascule de modèle depuis le panneau de paramètres.

    `switch_model` revérifie l'habilitation côté serveur : le contenu de
    ce payload vient du navigateur et ne fait pas foi.
    """
    assistant: llm | None = cl.user_session.get("assistant")
    if assistant is None:
        return

    demande = settings.get("model")
    if not demande or demande == assistant.model.model:
        return

    try:
        await assistant.switch_model(demande)
    except LLMAccessError as exc:
        await cl.Message(content=f"🔒 {exc}").send()
        return

    await cl.Message(
        content=f"Modèle basculé sur `{demande}`. La conversation est conservée."
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