"""Classe `ai_model` : dialogue avec un LLM via un client AsyncOpenAI.

Compatible aussi bien avec l'API OpenAI qu'avec un serveur exposant
l'interface OpenAI (Ollama sur /v1, vLLM, LM Studio, LiteLLM...).

Deux modes d'appel :
    - `initialyse_chat(message)`  -> réponse complète (str)
    - `stream_chat(message)`      -> générateur asynchrone de tokens
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DEFAULT_SYSTEM_PROMPT = (
    "Tu es l'assistant interne de la plateforme LLM souverain. "
    "Tu réponds en français, de façon concise et factuelle. "
    "Si tu ne sais pas, tu le dis."
)


class AIModelError(RuntimeError):
    """Erreur non récupérable après épuisement des tentatives."""


# =====================================================================
# Inventaire du serveur d'inférence
# =====================================================================
async def installed_models(base_url: str | None = None) -> set[str]:
    """Identifiants des modèles réellement chargés sur le serveur.

    Lève `AIModelError` si le serveur est injoignable : « aucun modèle
    installé » et « serveur éteint » appellent des messages différents,
    les confondre enverrait l'administrateur sur une fausse piste.
    """
    client = AsyncOpenAI(base_url=base_url or DEFAULT_OLLAMA_URL, api_key="ollama")
    try:
        page = await client.models.list()
        return {m.id for m in page.data}
    except openai.OpenAIError as exc:
        raise AIModelError(
            f"Serveur d'inférence injoignable sur {base_url or DEFAULT_OLLAMA_URL} : {exc}"
        ) from exc
    finally:
        await client.close()


def matches_installed(code_model: str, parc: set[str]) -> bool:
    """Règle de correspondance entre un code catalogué et le parc installé.

    Ollama nomme ses modèles « famille:tag ». Un code sans tag correspond
    à n'importe quel tag installé — « qwen2.5 » reconnaît
    « qwen2.5:latest », et le serveur résout de la même façon à l'appel.
    Avec un tag explicite, la comparaison est stricte : « llama3.2:3b »
    et « llama3.2:latest » sont des poids différents.

    Fonction unique et synchrone à dessein : catalogage, supervision et
    filtrage du sélecteur doivent appliquer exactement le même critère,
    sinon un modèle accepté à l'ajout se retrouve signalé manquant.
    """
    if code_model in parc:
        return True
    if ":" not in code_model:
        return any(m.split(":", 1)[0] == code_model for m in parc)
    return False


async def is_model_installed(code_model: str, base_url: str | None = None) -> bool:
    """Vrai si le modèle est chargé sur le serveur."""
    return matches_installed(code_model, await installed_models(base_url))


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0
    raw: Any = field(default=None, repr=False)

    def __str__(self) -> str:
        return self.content


# =====================================================================
class ai_model:
    """Un modèle + son client, avec l'historique d'une conversation."""

    def __init__(
        self,
        model: str,
        client: AsyncOpenAI,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.2, #0,7
        max_tokens: int | None = 1024,
        max_history: int = 20,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.model = model
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history = max_history  # nombre de messages hors system
        self.max_retries = max_retries
        self.timeout = timeout
        self.history: list[dict[str, str]] = []

    def __repr__(self) -> str:  # pragma: no cover
        return f"ai_model(model={self.model!r}, history={len(self.history)} messages)"

    # -----------------------------------------------------------------
    # Fabriques de client
    # -----------------------------------------------------------------
    @classmethod
    def from_ollama(
        cls,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        **kwargs: Any,
    ) -> "ai_model":
        """Ollama expose une API compatible OpenAI ; la clé est ignorée
        mais le SDK en exige une non vide."""
        client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        return cls(model=model, client=client, **kwargs)

    @classmethod
    def from_openai(cls, model: str, api_key: str | None = None, **kwargs: Any) -> "ai_model":
        client = AsyncOpenAI(api_key=api_key)  # à défaut, lit OPENAI_API_KEY
        return cls(model=model, client=client, **kwargs)

    # -----------------------------------------------------------------
    # Historique
    # -----------------------------------------------------------------
    def _build_messages(self, message: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})
        return messages

    def _remember(self, user_message: str, assistant_message: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        if len(self.history) > self.max_history:
            # On coupe par paires pour ne jamais laisser un tour orphelin.
            self.history = self.history[-self.max_history :]

    def reset(self, system_prompt: str | None = None) -> None:
        """Vide l'historique et, si fourni, change le prompt système."""
        self.history.clear()
        if system_prompt is not None:
            self.system_prompt = system_prompt

    # -----------------------------------------------------------------
    # Appel bloquant (réponse complète)
    # -----------------------------------------------------------------
    async def initialyse_chat(
        self,
        message: str,
        *,
        remember: bool = True,
        **overrides: Any,
    ) -> ChatResponse:
        """Envoie `message` au modèle et renvoie la réponse complète.

        `overrides` permet de surcharger ponctuellement temperature,
        max_tokens, ou tout autre paramètre de l'API.
        """
        messages = self._build_messages(message)
        started = time.perf_counter()

        response = await self._call_with_retry(
            messages=messages,
            stream=False,
            **overrides,
        )

        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)

        if remember:
            self._remember(message, content)

        result = ChatResponse(
            content=content,
            model=response.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=response,
        )
        logger.info(
            "Réponse de %s en %s ms (%s tokens générés)",
            result.model,
            result.latency_ms,
            result.completion_tokens,
        )
        return result

    # Alias sans faute de frappe, pour le nouveau code.
    initialise_chat = initialyse_chat

    # -----------------------------------------------------------------
    # Appel en streaming (pour Chainlit)
    # -----------------------------------------------------------------
    async def stream_chat(
        self,
        message: str,
        *,
        remember: bool = True,
        **overrides: Any,
    ) -> AsyncIterator[str]:
        """Génère la réponse token par token.

            async for token in model.stream_chat("Bonjour"):
                await msg.stream_token(token)
        """
        messages = self._build_messages(message)
        chunks: list[str] = []

        stream = await self._call_with_retry(messages=messages, stream=True, **overrides)

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    chunks.append(token)
                    yield token
        finally:
            if remember and chunks:
                self._remember(message, "".join(chunks))

    # -----------------------------------------------------------------
    # Appel bas niveau + politique de reprise
    # -----------------------------------------------------------------
    async def _call_with_retry(self, *, messages: list[dict[str, str]], stream: bool, **overrides: Any):
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        params.update(overrides)

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.client.chat.completions.create(**params),
                    timeout=self.timeout,
                )

            except (openai.APIConnectionError, openai.InternalServerError) as exc:
                last_error = exc
                logger.warning(
                    "Modèle injoignable (tentative %s/%s) : %s", attempt, self.max_retries, exc
                )

            except openai.RateLimitError as exc:
                last_error = exc
                logger.warning("Quota atteint (tentative %s/%s)", attempt, self.max_retries)

            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "Délai de %ss dépassé (tentative %s/%s)", self.timeout, attempt, self.max_retries
                )

            except openai.BadRequestError as exc:
                # Prompt trop long, modèle inconnu, paramètre invalide :
                # réessayer ne changera rien.
                raise AIModelError(f"Requête invalide pour le modèle '{self.model}' : {exc}") from exc

            except openai.AuthenticationError as exc:
                raise AIModelError(f"Authentification refusée : {exc}") from exc

            if attempt < self.max_retries:
                await asyncio.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s...

        raise AIModelError(
            f"Échec après {self.max_retries} tentatives sur '{self.model}' : {last_error}"
        ) from last_error

    # -----------------------------------------------------------------
    async def health_check(self) -> bool:
        """Vérifie que le modèle répond (utile au démarrage de l'app)."""
        try:
            await self.client.models.retrieve(self.model)
            return True
        except openai.OpenAIError as exc:
            logger.error("Modèle '%s' indisponible : %s", self.model, exc)
            return False

    async def close(self) -> None:
        await self.client.close()