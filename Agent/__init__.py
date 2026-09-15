"""Package Agent — trois implémentations derrière une interface commune."""

from .base import (
    AccesRefuse,
    AgentError,
    BaseAgent,
    BudgetEpuise,
    FournisseurModele,
    Etape,
    ReponseAgent,
    Source,
    TypeEtape,
)
from .loop_agentic import LoopAgentic
from .tools import Outil, RegistreOutils, outils_rag

__all__ = [
    "AccesRefuse", "AgentError", "BaseAgent", "BudgetEpuise", "Etape",
    "FournisseurModele", "ReponseAgent", "Source", "TypeEtape", "LoopAgentic",
    "Outil", "RegistreOutils", "outils_rag",
]


def _lazy(nom):
    """Import différé : pydantic-ai et langgraph sont optionnels.

    Le package doit rester utilisable avec le seul palier 1, qui
    n'ajoute aucune dépendance.
    """
    if nom == "PydanticAgent":
        from .pydantic_agent import PydanticAgent
        return PydanticAgent
    if nom == "LangGraphAgent":
        from .langgraph_agent import LangGraphAgent
        return LangGraphAgent
    raise AttributeError(nom)


def __getattr__(nom):  # PEP 562
    return _lazy(nom)
