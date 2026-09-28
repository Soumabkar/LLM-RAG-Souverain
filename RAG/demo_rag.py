"""Démonstration du RAG : python demo_rag.py

Indexe trois documents pour deux équipes, puis montre que chacune ne
retrouve que les siens — y compris à travers un agent.

Chaque composant bascule sur une variante locale s'il est indisponible :
MinIO → stockage en mémoire, Ollama → embeddings par hachage. La
plomberie est démontrée dans tous les cas ; seule la pertinence de la
recherche dépend d'un vrai modèle d'embedding.
"""

from __future__ import annotations

import asyncio
import logging
import os

from openai import AsyncOpenAI

from Engine.db import close_pool, get_connection
from RAG import (
    ConfigRAG,
    HachageEmbedder,
    MemoireStorage,
    MinIOStorage,
    MoteurRecherche,
    OllamaEmbedder,
    PgVectorStore,
    PipelineIndexation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
for bruyant in ("httpx", "httpx2", "openai", "openai._base_client"):
    logging.getLogger(bruyant).setLevel(logging.WARNING)
# Les nouvelles tentatives de connexion à MinIO sont attendues quand le
# service est arrêté : la démo bascule alors sur le stockage en mémoire.
logging.getLogger("urllib3").setLevel(logging.ERROR)

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DOCUMENTS = [
    ("conges.md", ["RH-01"], (
        "# Procédure de congés\n\n"
        "Les demandes de congés se font dans l'outil RH au moins deux semaines à "
        "l'avance. Le manager valide sous cinq jours ouvrés. Pour toute question, "
        "écrire à rh@entreprise.fr ou appeler le 01 23 45 67 89."
    )),
    ("lakehouse.md", ["DATA-01"], (
        "# Architecture lakehouse\n\n"
        "Le stockage objet MinIO héberge les fichiers au format Parquet. Trino "
        "interroge le catalogue Hive Metastore pour localiser les tables, puis lit "
        "directement les fichiers. PostgreSQL ne stocke que les métadonnées."
    )),
    ("clients.csv", ["DATA-01", "RH-01"], (
        "nom;ville;statut\nDupont;Paris;actif\nMartin;Lyon;inactif\nBernard;Nantes;actif"
    )),
]


def titre(texte: str) -> None:
    print(f"\n{'─' * 72}\n  {texte}\n{'─' * 72}")


# =====================================================================
async def construire_stockage(config: ConfigRAG):
    stockage = MinIOStorage(config.bucket)
    try:
        await asyncio.wait_for(stockage.assurer_bucket(), timeout=5)
        print(f"MinIO joignable — bucket « {config.bucket} »")
        return stockage
    except Exception as exc:  # noqa: BLE001
        print(f"MinIO injoignable ({type(exc).__name__}) — stockage en mémoire")
        return MemoireStorage(config.bucket)


async def construire_embedder(config: ConfigRAG):
    client = AsyncOpenAI(base_url=OLLAMA, api_key="ollama", max_retries=0, timeout=10)
    embedder = OllamaEmbedder(client, config.modele_embedding, config.dimension)
    try:
        await embedder.requete("test")
        print(f"Ollama joignable — embeddings « {config.modele_embedding} »")
        return embedder
    except Exception as exc:  # noqa: BLE001
        print(f"Ollama injoignable ({type(exc).__name__}) — embeddings simulés par hachage")
        return HachageEmbedder(config.dimension)


# =====================================================================
async def main() -> None:
    config = ConfigRAG.depuis_env()
    stockage = await construire_stockage(config)
    embedder = await construire_embedder(config)
    base = PgVectorStore(get_connection)

    pipeline = PipelineIndexation(stockage, embedder, base, config)
    await pipeline.preparer()

    # -----------------------------------------------------------------
    titre("A. Indexation hors ligne — les quatre étapes")
    for nom, equipes, contenu in DOCUMENTS:
        bilan = await pipeline.indexer(contenu.encode("utf-8"), nom, equipes)
        s = bilan.securisation
        print(f"  {nom:<14} {bilan.document.doc_id:<28} équipes={equipes}")
        print(f"  {'':<14} {bilan.caracteres_extraits} car. extraits · "
              f"{s.total_masque} donnée(s) personnelle(s) masquée(s) · "
              f"{bilan.chunks} chunk(s) · {bilan.vecteurs} vecteur(s)")

    doc_rh = next(d for d in await pipeline.documents() if d.nom == "conges.md")
    corpus = await stockage.lire_texte(f"{config.prefixe_corpus}/{doc_rh.doc_id}.md")
    phrase = next(p for p in corpus.split(". ") if "[EMAIL]" in p or "[TELEPHONE]" in p)
    print(f"\n  Corpus sécurisé de conges.md :\n    « {phrase.strip()} »")

    # -----------------------------------------------------------------
    titre("B. Recherche — chaque équipe ne voit que ses documents")
    moteur = MoteurRecherche(embedder, base)
    question = "Comment poser des congés ?"

    for equipes in (["RH-01"], ["DATA-01"], []):
        resultats = await moteur.recherche_semantique(question, top_k=3, equipes=equipes)
        etiquette = ", ".join(equipes) or "aucune équipe"
        sources = sorted({r.source for r in resultats}) or ["— rien —"]
        print(f"  {etiquette:<16} → {', '.join(sources)}")

    # -----------------------------------------------------------------
    titre("C. Outil pour les agents — périmètre figé")
    outil = moteur.outils_pour_equipes(["DATA-01"]).get("recherche_documentaire")
    print(f"  arguments exposés au modèle : "
          f"{list(outil.schema_openai()['function']['parameters']['properties'])}")
    try:
        await outil.appeler({"requete": "congés", "equipes": ["RH-01"]})
    except ValueError as exc:
        print(f"  tentative d'élargissement refusée : {str(exc)[:80]}…")

    # -----------------------------------------------------------------
    titre("D. Droit à l'effacement")
    retires = await pipeline.supprimer(doc_rh.doc_id)
    restants = await moteur.recherche_semantique(question, top_k=5, equipes=["RH-01"])
    print(f"  conges.md supprimé : {retires} vecteur(s) retiré(s)")
    print(f"  RH-01 retrouve encore : {sorted({r.source for r in restants}) or 'rien'}")

    # Nettoyage de la démonstration
    for document in await pipeline.documents():
        await pipeline.supprimer(document.doc_id)
    await close_pool()
    print("\nDémonstration terminée, documents retirés.")


if __name__ == "__main__":
    asyncio.run(main())
