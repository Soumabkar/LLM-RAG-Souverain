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
from LLM.llm import LLMAccessError, LLMModelUnavailable, llm
from Engine.models import authenticate, user

logging.basicConfig(level=logging.INFO)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")


def bloc_equipes(equipes: list[dict]) -> str:
    """Tableau markdown des équipes de l'utilisateur.

    Les équipes sans modèle actif sont affichées quand même : c'est
    souvent l'explication d'une liste de modèles plus courte que prévu.
    """
    if not equipes:
        return (
            "**Vous n'appartenez à aucune équipe.**\n\n"
            "L'accès aux modèles passe par l'équipe : demandez à un "
            "administrateur de vous rattacher."
        )

    lignes = [
        f"**Vos équipes ({len(equipes)})**",
        "",
        "| Équipe | Code | Modèles accessibles |",
        "| --- | --- | --- |",
    ]
    for e in equipes:
        modeles = ", ".join(e["models"]) if e["models"] else "_aucun modèle actif_"
        lignes.append(f"| {e['team_name']} | `{e['code_team']}` | {modeles} |")

    total = len({m for e in equipes for m in e["models"]})
    lignes += ["", f"Soit **{total} modèle(s)** au total, un modèle partagé "
                   "par deux équipes n'étant compté qu'une fois."]
    return "\n".join(lignes)


@cl.action_callback("mes_equipes")
async def afficher_equipes(action: cl.Action) -> None:
    """Réaffiche la liste à la demande, sans recharger la page."""
    assistant: llm | None = cl.user_session.get("assistant")
    if assistant is None:
        return
    await cl.Message(content=bloc_equipes(await assistant.user.teams_detail())).send()


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

    # Intersection entre les habilitations de l'utilisateur et le parc
    # réellement installé : un modèle catalogué mais absent du serveur
    # apparaîtrait dans le sélecteur et renverrait une 404 au premier message.
    modeles = await assistant.available_models()
    if not modeles:
        habilites = await compte.allowed_models()
        detail = (
            "Demande à un administrateur de rattacher ton équipe à un modèle."
            if not habilites
            else (
                "Les modèles autorisés pour toi ne sont pas installés sur le "
                "serveur d'inférence : "
                + ", ".join(f"`{m['code_model']}`" for m in habilites)
            )
        )
        await cl.Message(content=f"Aucun modèle disponible pour `{compte.email}`. {detail}").send()
        return

    # Modèle d'ouverture : celui du .env s'il est disponible, sinon le premier.
    codes = [m["code_model"] for m in modeles]
    model.model = DEFAULT_MODEL if DEFAULT_MODEL in codes else codes[0]

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

    courant = next(m["display_name"] for m in modeles if m["code_model"] == model.model)
    equipes = await compte.teams_detail()

    await cl.Message(
        content=(
            f"Bonjour **{compte.login}**.\n\n"
            f"{bloc_equipes(equipes)}\n\n"
            f"Modèle actif : **{courant}** — {len(modeles)} disponible(s) "
            "dans les paramètres."
        ),
        actions=[
            cl.Action(name="mes_equipes", payload={}, label="Mes équipes",
                      tooltip="Réafficher vos équipes et leurs modèles"),
        ],
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
    except LLMModelUnavailable as exc:
        await cl.Message(content=f"⚠️ {exc}").send()
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
    except LLMModelUnavailable as exc:
        reply.content = f"⚠️ {exc}"
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