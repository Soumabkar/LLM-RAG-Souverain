"""Stockage objet.

`MinIOStorage` parle le protocole S3 : le même code fonctionne sur MinIO
en local, et sur Amazon S3, GCS en mode interopérable ou Azure via une
passerelle le jour d'une migration.

Le SDK MinIO est synchrone : chaque appel est déporté dans un fil par
`asyncio.to_thread`, pour ne pas bloquer la boucle d'événements de
Chainlit pendant un téléversement.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StockageError(RuntimeError):
    """Objet introuvable, bucket inaccessible ou identifiants refusés."""


class Stockage(ABC):
    """Contrat minimal d'un stockage objet."""

    @abstractmethod
    async def assurer_bucket(self) -> None: ...

    @abstractmethod
    async def ecrire(self, cle: str, donnees: bytes, type_mime: str = "application/octet-stream") -> None: ...

    @abstractmethod
    async def lire(self, cle: str) -> bytes: ...

    @abstractmethod
    async def lister(self, prefixe: str) -> list[str]: ...

    @abstractmethod
    async def existe(self, cle: str) -> bool: ...

    @abstractmethod
    async def supprimer(self, cle: str) -> None: ...

    async def ecrire_texte(self, cle: str, texte: str) -> None:
        await self.ecrire(cle, texte.encode("utf-8"), "text/plain; charset=utf-8")

    async def lire_texte(self, cle: str) -> str:
        return (await self.lire(cle)).decode("utf-8")


# =====================================================================
class MinIOStorage(Stockage):
    """Stockage sur MinIO ou tout service compatible S3."""

    def __init__(
        self,
        bucket: str,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool | None = None,
        tentatives: int = 2,
        delai_s: float = 5.0,
    ):
        import urllib3
        from minio import Minio

        self.bucket = bucket
        # Par défaut, le SDK réessaie cinq fois avec attente exponentielle :
        # un MinIO arrêté fait patienter près d'une minute avant l'erreur.
        # Deux tentatives et un délai court suffisent sur un réseau local.
        http = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=delai_s, read=delai_s * 6),
            retries=urllib3.Retry(total=tentatives, backoff_factor=0.2,
                                  status_forcelist=[500, 502, 503, 504]),
        )
        self._client = Minio(
            (endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")).strip(),
            access_key=(access_key or os.getenv("MINIO_ACCESS_KEY", "")).strip(),
            secret_key=(secret_key or os.getenv("MINIO_SECRET_KEY", "")).strip(),
            secure=(os.getenv("MINIO_SECURE", "false").strip().lower() == "true")
            if secure is None else secure,
            http_client=http,
        )

    async def assurer_bucket(self) -> None:
        def _creer():
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
                logger.info("Bucket %s créé", self.bucket)
        await asyncio.to_thread(_creer)

    async def ecrire(self, cle: str, donnees: bytes, type_mime: str = "application/octet-stream") -> None:
        await asyncio.to_thread(
            self._client.put_object, self.bucket, cle,
            io.BytesIO(donnees), len(donnees), content_type=type_mime,
        )

    async def lire(self, cle: str) -> bytes:
        def _lire() -> bytes:
            from minio.error import S3Error
            try:
                reponse = self._client.get_object(self.bucket, cle)
            except S3Error as exc:
                raise StockageError(f"Objet '{cle}' illisible : {exc.code}") from exc
            try:
                return reponse.read()
            finally:
                reponse.close()
                reponse.release_conn()
        return await asyncio.to_thread(_lire)

    async def lister(self, prefixe: str) -> list[str]:
        def _lister() -> list[str]:
            return sorted(
                o.object_name
                for o in self._client.list_objects(self.bucket, prefix=prefixe, recursive=True)
            )
        return await asyncio.to_thread(_lister)

    async def existe(self, cle: str) -> bool:
        def _existe() -> bool:
            from minio.error import S3Error
            try:
                self._client.stat_object(self.bucket, cle)
                return True
            except S3Error:
                return False
        return await asyncio.to_thread(_existe)

    async def supprimer(self, cle: str) -> None:
        await asyncio.to_thread(self._client.remove_object, self.bucket, cle)


# =====================================================================
class MemoireStorage(Stockage):
    """Stockage en mémoire — pour les tests et le développement.

    Même contrat que `MinIOStorage` : un pipeline testé ici fonctionne
    tel quel sur MinIO.
    """

    def __init__(self, bucket: str = "memoire"):
        self.bucket = bucket
        self._objets: dict[str, bytes] = {}

    async def assurer_bucket(self) -> None:
        return None

    async def ecrire(self, cle: str, donnees: bytes, type_mime: str = "application/octet-stream") -> None:
        self._objets[cle] = bytes(donnees)

    async def lire(self, cle: str) -> bytes:
        if cle not in self._objets:
            raise StockageError(f"Objet '{cle}' introuvable")
        return self._objets[cle]

    async def lister(self, prefixe: str) -> list[str]:
        return sorted(c for c in self._objets if c.startswith(prefixe))

    async def existe(self, cle: str) -> bool:
        return cle in self._objets

    async def supprimer(self, cle: str) -> None:
        self._objets.pop(cle, None)
