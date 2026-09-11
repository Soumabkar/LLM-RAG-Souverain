"""Contrôle d'accès aux modèles au niveau de la classe `llm`."""

from __future__ import annotations

import types

import pytest

from Model.ai_model import ai_model
from LLM.llm import LLMAccessError, llm
from Engine.models import user

pytestmark = pytest.mark.db


class _FauxCompletions:
    async def create(self, **kw):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=f"réponse de {kw['model']}")
                )
            ],
            model=kw["model"],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _FauxClient:
    chat = types.SimpleNamespace(completions=_FauxCompletions())

    async def close(self) -> None:
        pass


def session(compte: user, code_model: str = "llama3.1:8b", **kw) -> llm:
    return llm(ai_model(code_model, _FauxClient()), compte, **kw)


# =====================================================================
class TestAutorisationInitiale:
    async def test_modele_autorise(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        assert await session(karim).response_llm("bonjour")

    async def test_modele_non_autorise(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        with pytest.raises(LLMAccessError, match="mistral"):
            await session(karim, "mistral:7b").response_llm("bonjour")

    async def test_sans_aucune_habilitation(self, catalogue, karim):
        with pytest.raises(LLMAccessError):
            await session(karim).response_llm("bonjour")


# =====================================================================
class TestChangementDeModele:
    async def test_bascule_autorisee(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        await equipe_data.add_model("mistral:7b")

        assistant = session(karim)
        await assistant.response_llm("premier")
        await assistant.switch_model("mistral:7b")

        assert assistant.model.model == "mistral:7b"
        assert "mistral:7b" in await assistant.response_llm("second")

    async def test_historique_conserve(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        await equipe_data.add_model("mistral:7b")

        assistant = session(karim)
        await assistant.response_llm("premier")
        await assistant.switch_model("mistral:7b")

        assert len(assistant.model.history) == 2

    async def test_bascule_refusee(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        assistant = session(karim)
        await assistant.response_llm("bonjour")

        with pytest.raises(LLMAccessError, match="mistral"):
            await assistant.switch_model("mistral:7b")
        assert assistant.model.model == "llama3.1:8b"

    async def test_bascule_vers_le_meme_modele(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        assistant = session(karim)
        await assistant.switch_model("llama3.1:8b")
        assert assistant.model.model == "llama3.1:8b"

    async def test_modele_hors_catalogue(self, catalogue, equipe_data, karim):
        await equipe_data.add_model("llama3.1:8b")
        assistant = session(karim)
        with pytest.raises(LLMAccessError):
            await assistant.switch_model("nexiste:pas")

    async def test_habilitation_revoquee_entre_deux_bascules(
        self, catalogue, equipe_data, karim
    ):
        """Le contrôle est refait à chaque bascule, pas mis en cache."""
        await equipe_data.add_model("llama3.1:8b")
        await equipe_data.add_model("mistral:7b")

        assistant = session(karim)
        await assistant.switch_model("mistral:7b")
        await equipe_data.remove_model("llama3.1:8b")

        with pytest.raises(LLMAccessError):
            await assistant.switch_model("llama3.1:8b")

    async def test_available_models(self, catalogue, equipe_data, equipe_secu, karim):
        await karim.join_team("SEC-01")
        await equipe_data.add_model("llama3.1:8b")
        await equipe_secu.add_model("mistral:7b")

        codes = [m["code_model"] for m in await session(karim).available_models()]
        assert codes == ["llama3.1:8b", "mistral:7b"]
