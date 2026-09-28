"""Bases vectorielles.

Trois implémentations derrière une interface commune :

| Base      | Quand la choisir |
|-----------|------------------|
| pgvector  | **Par défaut** — réutilise le PostgreSQL existant, filtrage d'accès en SQL |
| Qdrant    | Volumes importants, filtres riches, service dédié |
| FAISS     | Développement et tests, sans serveur |

Toutes filtrent par équipe **avant** de rendre les résultats : un
utilisateur ne voit que les extraits des documents de ses équipes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager, Callable

from .config import Chunk, ResultatRecherche

logger = logging.getLogger(__name__)


class BaseVectorielle(ABC):
    nom: str = "base"

    @abstractmethod
    async def creer(self, dimension: int) -> None:
        """Crée la collection si elle n'existe pas."""

    @abstractmethod
    async def inserer(self, chunks: list[Chunk], vecteurs: list[list[float]]) -> int:
        """Insère ou remplace. Renvoie le nombre de vecteurs écrits."""

    @abstractmethod
    async def rechercher(
        self, vecteur: list[float], top_k: int = 5, equipes: list[str] | None = None
    ) -> list[ResultatRecherche]:
        """Plus proches voisins, restreints aux équipes fournies.

        `equipes=None` désactive le filtre — réservé aux traitements
        d'administration, jamais à une requête utilisateur.
        """

    @abstractmethod
    async def supprimer_document(self, doc_id: str) -> int:
        """Retire tous les chunks d'un document."""

    @staticmethod
    def _verifier(chunks: list[Chunk], vecteurs: list[list[float]]) -> None:
        if len(chunks) != len(vecteurs):
            raise ValueError(f"{len(chunks)} chunks pour {len(vecteurs)} vecteurs.")


# =====================================================================
# pgvector
# =====================================================================
class PgVectorStore(BaseVectorielle):
    """Vecteurs dans le PostgreSQL du projet.

    Prérequis : l'extension `vector`. L'image `postgres:16-alpine` ne
    l'embarque pas — passer à `pgvector/pgvector:pg16` dans le compose.

    Le contrôle d'accès est une simple clause SQL `equipes && %s`
    (intersection de tableaux) : c'est l'argument décisif de pgvector
    ici, le filtre et la recherche sont dans la même requête.
    """

    nom = "pgvector"

    def __init__(self, connexion: Callable[[], AsyncContextManager] | None = None,
                 table: str = "llm_souverain.rag_chunk"):
        if connexion is None:
            from Engine.db import get_connection as connexion  # type: ignore
        self._connexion = connexion
        self.table = table

    @staticmethod
    def _litteral(vecteur: list[float]) -> str:
        # Format texte de pgvector : évite une dépendance au paquet
        # `pgvector` côté Python.
        return "[" + ",".join(f"{x:.7f}" for x in vecteur) + "]"

    async def creer(self, dimension: int) -> None:
        async with self._connexion() as conn, conn.cursor() as cur:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id   VARCHAR(128) PRIMARY KEY,
                    doc_id     VARCHAR(64)  NOT NULL,
                    idx        INTEGER      NOT NULL,
                    texte      TEXT         NOT NULL,
                    source     VARCHAR(512) NOT NULL,
                    type_doc   VARCHAR(16)  NOT NULL,
                    equipes    TEXT[]       NOT NULL DEFAULT '{{}}',
                    embedding  vector({dimension}) NOT NULL,
                    indexe_le  TIMESTAMPTZ  NOT NULL DEFAULT now()
                )
            """)
            nom = self.table.split(".")[-1]
            # HNSW : recherche approximative en temps sous-linéaire.
            await cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{nom}_embedding
                    ON {self.table} USING hnsw (embedding vector_cosine_ops)
            """)
            await cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{nom}_equipes ON {self.table} USING gin (equipes)"
            )
            await cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{nom}_doc ON {self.table} (doc_id)"
            )

    async def inserer(self, chunks: list[Chunk], vecteurs: list[list[float]]) -> int:
        self._verifier(chunks, vecteurs)
        if not chunks:
            return 0
        async with self._connexion() as conn, conn.cursor() as cur:
            await cur.executemany(
                f"""
                INSERT INTO {self.table}
                    (chunk_id, doc_id, idx, texte, source, type_doc, equipes, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    texte = EXCLUDED.texte,
                    equipes = EXCLUDED.equipes,
                    embedding = EXCLUDED.embedding,
                    indexe_le = now()
                """,
                [
                    (c.chunk_id, c.doc_id, c.index, c.texte, c.source, c.type.value,
                     c.equipes, self._litteral(v))
                    for c, v in zip(chunks, vecteurs)
                ],
            )
        return len(chunks)

    async def rechercher(self, vecteur, top_k=5, equipes=None):
        filtre = "WHERE equipes && %s::text[]" if equipes is not None else ""
        params: list[Any] = [self._litteral(vecteur)]
        if equipes is not None:
            params.append(list(equipes))
        params += [self._litteral(vecteur), top_k]

        async with self._connexion() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT chunk_id, doc_id, texte, source, equipes,
                       1 - (embedding <=> %s::vector) AS score
                  FROM {self.table}
                  {filtre}
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s
                """,
                params,
            )
            lignes = await cur.fetchall()

        return [
            ResultatRecherche(
                chunk_id=l["chunk_id"], doc_id=l["doc_id"], texte=l["texte"],
                source=l["source"], score=float(l["score"]), equipes=list(l["equipes"]),
            )
            for l in lignes
        ]

    async def supprimer_document(self, doc_id: str) -> int:
        async with self._connexion() as conn, conn.cursor() as cur:
            await cur.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))
            return cur.rowcount


# =====================================================================
# FAISS
# =====================================================================
class FaissStore(BaseVectorielle):
    """Index FAISS en mémoire, persistable sur disque.

    FAISS ne sait pas filtrer sur des métadonnées : on sur-échantillonne
    puis on filtre. Correct pour quelques milliers de chunks ; au-delà,
    un filtre très sélectif peut rendre moins de `top_k` résultats.
    C'est pourquoi FAISS reste ici l'option de développement.
    """

    nom = "faiss"
    FACTEUR_SURECHANTILLON = 8

    def __init__(self):
        self._index = None
        self._metas: list[Chunk] = []
        self._dimension = 0

    async def creer(self, dimension: int) -> None:
        import faiss

        if self._index is None:
            # Produit scalaire sur vecteurs normalisés = similarité cosinus.
            self._index = faiss.IndexFlatIP(dimension)
            self._dimension = dimension

    @staticmethod
    def _normaliser(vecteurs):
        import numpy as np
        m = np.asarray(vecteurs, dtype="float32")
        normes = np.linalg.norm(m, axis=1, keepdims=True)
        normes[normes == 0] = 1.0
        return m / normes

    async def inserer(self, chunks, vecteurs):
        self._verifier(chunks, vecteurs)
        if not chunks:
            return 0
        # Remplacement : on retire d'abord les chunks déjà présents.
        ids = {c.chunk_id for c in chunks}
        if any(m.chunk_id in ids for m in self._metas):
            await self._reconstruire([m for m in self._metas if m.chunk_id not in ids])
        self._index.add(self._normaliser(vecteurs))
        self._metas.extend(chunks)
        return len(chunks)

    async def _reconstruire(self, gardes: list[Chunk]) -> None:
        # IndexFlat ne supporte pas bien la suppression : on reconstruit.
        # Les vecteurs sont conservés dans l'index, on les relit.
        import faiss
        anciens = {m.chunk_id: i for i, m in enumerate(self._metas)}
        vecteurs = [self._index.reconstruct(anciens[g.chunk_id]) for g in gardes]
        self._index = faiss.IndexFlatIP(self._dimension)
        if vecteurs:
            self._index.add(self._normaliser(vecteurs))
        self._metas = list(gardes)

    async def rechercher(self, vecteur, top_k=5, equipes=None):
        if self._index is None or not self._metas:
            return []
        k = min(len(self._metas), top_k * (self.FACTEUR_SURECHANTILLON if equipes is not None else 1))
        scores, positions = self._index.search(self._normaliser([vecteur]), k)
        autorisees = set(equipes) if equipes is not None else None

        resultats = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0:
                continue
            meta = self._metas[pos]
            if autorisees is not None and not autorisees.intersection(meta.equipes):
                continue
            resultats.append(ResultatRecherche(
                chunk_id=meta.chunk_id, doc_id=meta.doc_id, texte=meta.texte,
                source=meta.source, score=float(score), equipes=meta.equipes,
            ))
            if len(resultats) == top_k:
                break
        return resultats

    async def supprimer_document(self, doc_id: str) -> int:
        avant = len(self._metas)
        await self._reconstruire([m for m in self._metas if m.doc_id != doc_id])
        return avant - len(self._metas)


# =====================================================================
# Qdrant
# =====================================================================
class QdrantStore(BaseVectorielle):
    """Collection Qdrant.

    `QdrantClient(":memory:")` ou un chemin local fonctionnent sans
    serveur — pratique en test. En production : `url="http://qdrant:6333"`.
    Le filtre par équipe est un filtre de charge utile natif, appliqué
    pendant la recherche et non après.
    """

    nom = "qdrant"

    def __init__(self, client: Any = None, collection: str = "rag_souverain"):
        from qdrant_client import QdrantClient
        self.client = client or QdrantClient(":memory:")
        self.collection = collection

    @staticmethod
    def _id(chunk_id: str) -> str:
        # Qdrant exige un entier ou un UUID : on dérive un UUID stable.
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    async def creer(self, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        def _creer():
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    self.collection,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
        await asyncio.to_thread(_creer)

    async def inserer(self, chunks, vecteurs):
        from qdrant_client.models import PointStruct
        self._verifier(chunks, vecteurs)
        if not chunks:
            return 0
        points = [
            PointStruct(id=self._id(c.chunk_id), vector=v, payload=json.loads(c.model_dump_json()))
            for c, v in zip(chunks, vecteurs)
        ]
        await asyncio.to_thread(self.client.upsert, self.collection, points)
        return len(points)

    async def rechercher(self, vecteur, top_k=5, equipes=None):
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        filtre = None
        if equipes is not None:
            filtre = Filter(must=[FieldCondition(key="equipes", match=MatchAny(any=list(equipes)))])

        def _chercher():
            return self.client.query_points(
                self.collection, query=vecteur, limit=top_k, query_filter=filtre,
            ).points

        points = await asyncio.to_thread(_chercher)
        return [
            ResultatRecherche(
                chunk_id=p.payload["chunk_id"], doc_id=p.payload["doc_id"],
                texte=p.payload["texte"], source=p.payload["source"],
                score=float(p.score), equipes=p.payload.get("equipes", []),
            )
            for p in points
        ]

    async def supprimer_document(self, doc_id: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        filtre = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

        def _supprimer():
            avant = self.client.count(self.collection, count_filter=filtre).count
            self.client.delete(self.collection, points_selector=FilterSelector(filter=filtre))
            return avant

        return await asyncio.to_thread(_supprimer)
