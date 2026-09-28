"""Pipeline d'indexation hors ligne — le bloc (A) du schéma.

    store_data_brute  →  data_processing  →  data_chunking  →  insertVectorDB
         ①                   ②                  ③                  ④

Chaque étape lit la sortie de la précédente **dans le stockage**, pas en
mémoire. Conséquence voulue : n'importe quelle étape peut être rejouée
seule. Changer la taille des chunks ne demande pas de réextraire les
PDF ; changer de modèle d'embedding ne demande pas de redécouper.

Arborescence du bucket :

    rag-souverain/
    ├── data-brute/{type}/{doc_id}/{fichier}      source inchangée
    ├── data-brute/{type}/{doc_id}/document.json  métadonnées, équipes
    ├── data-processing/{doc_id}.txt              texte extrait, avant nettoyage
    ├── data-corpus/{doc_id}.md                   texte nettoyé et sécurisé
    ├── data-corpus/{doc_id}.json                 rapport de sécurisation
    └── data-chunking/{doc_id}.jsonl              extraits prêts à vectoriser
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import re

from .chunking import decouper
from .config import (
    EXTENSIONS,
    BilanIndexation,
    Chunk,
    ConfigRAG,
    DocumentBrut,
    RapportSecurisation,
    TypeDocument,
)
from .embeddings import Embedder
from .extraction import extraire_texte
from .processing import traiter
from .storage import Stockage
from .vectorstores import BaseVectorielle

logger = logging.getLogger(__name__)

TAILLE_MAX_OCTETS = 100 * 1024 * 1024  # 100 Mo par document


class IngestionError(ValueError):
    """Document refusé à l'ingestion."""


class PipelineIndexation:
    """Les quatre étapes de l'indexation, adossées au stockage objet."""

    def __init__(
        self,
        stockage: Stockage,
        embedder: Embedder,
        base: BaseVectorielle,
        config: ConfigRAG | None = None,
    ):
        self.stockage = stockage
        self.embedder = embedder
        self.base = base
        self.config = config or ConfigRAG()

    # -----------------------------------------------------------------
    async def preparer(self) -> None:
        """Crée le bucket et la collection vectorielle s'ils manquent."""
        await self.stockage.assurer_bucket()
        await self.base.creer(self.embedder.dimension)

    # -----------------------------------------------------------------
    # Clés de stockage
    # -----------------------------------------------------------------
    @staticmethod
    def _type_depuis_id(doc_id: str) -> TypeDocument:
        return TypeDocument(doc_id.split("-", 1)[0])

    def _cle_meta(self, doc_id: str) -> str:
        return f"{self.config.prefixe_brute}/{self._type_depuis_id(doc_id).value}/{doc_id}/document.json"

    def _cle_processing(self, doc_id: str) -> str:
        return f"{self.config.prefixe_processing}/{doc_id}.txt"

    def _cle_corpus(self, doc_id: str) -> str:
        return f"{self.config.prefixe_corpus}/{doc_id}.md"

    def _cle_rapport(self, doc_id: str) -> str:
        return f"{self.config.prefixe_corpus}/{doc_id}.json"

    def _cle_chunks(self, doc_id: str) -> str:
        return f"{self.config.prefixe_chunking}/{doc_id}.jsonl"

    @staticmethod
    def _nom_sur(nom_fichier: str) -> str:
        """Nom de fichier sans chemin ni caractère exotique.

        Sans ce contrôle, un fichier nommé « ../../data-corpus/x.md »
        écrirait hors de sa zone : la traversée de chemin existe aussi
        dans un stockage objet.
        """
        nom = posixpath.basename(nom_fichier.replace("\\", "/"))
        nom = re.sub(r"[^A-Za-z0-9._-]", "_", nom).strip("._")
        if not nom:
            raise IngestionError("Nom de fichier vide après nettoyage.")
        return nom[:200]

    async def _document(self, doc_id: str) -> DocumentBrut:
        return DocumentBrut.model_validate_json(await self.stockage.lire(self._cle_meta(doc_id)))

    # =================================================================
    # ① Stockage brut
    # =================================================================
    async def store_data_brute(
        self,
        contenu: bytes,
        nom_fichier: str,
        equipes: list[str],
        type_doc: TypeDocument | None = None,
    ) -> DocumentBrut:
        """Dépose un fichier en zone brute, rangé par type.

        L'identifiant dérive du contenu (SHA-256) : déposer deux fois le
        même fichier ne crée pas de doublon, et les équipes du dernier
        dépôt font foi.

        Args:
            equipes: équipes autorisées à retrouver ce document. Obligatoire —
                un document sans équipe ne serait visible de personne.
        """
        if not contenu:
            raise IngestionError("Fichier vide.")
        if len(contenu) > TAILLE_MAX_OCTETS:
            raise IngestionError(
                f"Fichier de {len(contenu) // (1024 * 1024)} Mo, "
                f"limite {TAILLE_MAX_OCTETS // (1024 * 1024)} Mo."
            )
        if not equipes:
            raise IngestionError(
                "Au moins une équipe est requise : un document sans équipe "
                "ne serait retrouvable par personne."
            )

        nom = self._nom_sur(nom_fichier)
        if type_doc is None:
            extension = posixpath.splitext(nom)[1].lower()
            if extension not in EXTENSIONS:
                raise IngestionError(
                    f"Format '{extension or '(sans extension)'}' non pris en charge. "
                    f"Acceptés : {', '.join(sorted(EXTENSIONS))}."
                )
            type_doc = EXTENSIONS[extension]

        sha = hashlib.sha256(contenu).hexdigest()
        doc_id = f"{type_doc.value}-{sha[:16]}"
        cle = f"{self.config.prefixe_brute}/{type_doc.value}/{doc_id}/{nom}"

        document = DocumentBrut(
            doc_id=doc_id, nom=nom, type=type_doc, cle=cle,
            equipes=sorted(set(equipes)), sha256=sha, taille=len(contenu),
        )
        await self.stockage.ecrire(cle, contenu)
        await self.stockage.ecrire(self._cle_meta(doc_id), document.model_dump_json(indent=2).encode(),
                                   "application/json")
        logger.info("Brut : %s (%s, %d octets, équipes %s)", doc_id, nom, len(contenu), equipes)
        return document

    # =================================================================
    # ② Traitement
    # =================================================================
    async def data_processing(self, doc_id: str) -> tuple[int, RapportSecurisation]:
        """Extrait le texte, le nettoie et masque les données personnelles.

        Deux sorties : le texte extrait tel quel (`data-processing`), pour
        diagnostiquer une extraction défaillante, et le texte sécurisé
        (`data-corpus`), seul autorisé à entrer dans la base vectorielle.

        Returns:
            Nombre de caractères extraits, et rapport de sécurisation.
        """
        document = await self._document(doc_id)
        brut = await self.stockage.lire(document.cle)

        texte = extraire_texte(brut, document.type)
        await self.stockage.ecrire_texte(self._cle_processing(doc_id), texte)

        corpus, rapport = traiter(texte)
        await self.stockage.ecrire_texte(self._cle_corpus(doc_id), corpus)
        await self.stockage.ecrire(
            self._cle_rapport(doc_id),
            json.dumps({"doc_id": doc_id, **rapport.model_dump()}, indent=2).encode(),
            "application/json",
        )
        if rapport.total_masque:
            logger.info("Traitement %s : %d donnée(s) personnelle(s) masquée(s)",
                        doc_id, rapport.total_masque)
        return len(texte), rapport

    # =================================================================
    # ③ Découpage
    # =================================================================
    async def data_chunking(self, doc_id: str) -> list[Chunk]:
        """Découpe le corpus sécurisé en extraits."""
        document = await self._document(doc_id)
        corpus = await self.stockage.lire_texte(self._cle_corpus(doc_id))

        chunks = decouper(
            corpus, doc_id=doc_id, source=document.nom, type_doc=document.type,
            equipes=document.equipes, taille=self.config.taille_chunk,
            chevauchement=self.config.chevauchement,
        )
        lignes = "\n".join(c.model_dump_json() for c in chunks)
        await self.stockage.ecrire(self._cle_chunks(doc_id), lignes.encode("utf-8"),
                                   "application/x-ndjson")
        return chunks

    # =================================================================
    # ④ Indexation vectorielle
    # =================================================================
    async def insertVectorDB(self, doc_id: str) -> int:  # noqa: N802 — nom imposé
        """Vectorise les extraits et les écrit dans la base.

        Les anciens chunks du document sont d'abord supprimés : si un
        retraitement produit moins d'extraits qu'avant, les surnuméraires
        ne doivent pas rester en base et continuer à remonter.
        """
        donnees = await self.stockage.lire_texte(self._cle_chunks(doc_id))
        chunks = [Chunk.model_validate_json(l) for l in donnees.splitlines() if l.strip()]
        if not chunks:
            logger.warning("Aucun chunk pour %s — document vide après traitement ?", doc_id)
            return 0

        vecteurs = await self.embedder.documents([c.texte for c in chunks])
        await self.base.supprimer_document(doc_id)
        ecrits = await self.base.inserer(chunks, vecteurs)
        logger.info("Indexation %s : %d vecteur(s) dans %s", doc_id, ecrits, self.base.nom)
        return ecrits

    insert_vector_db = insertVectorDB  # alias au format Python

    # =================================================================
    async def indexer(
        self, contenu: bytes, nom_fichier: str, equipes: list[str],
        type_doc: TypeDocument | None = None,
    ) -> BilanIndexation:
        """Les quatre étapes enchaînées."""
        document = await self.store_data_brute(contenu, nom_fichier, equipes, type_doc)
        extraits, rapport = await self.data_processing(document.doc_id)
        chunks = await self.data_chunking(document.doc_id)
        vecteurs = await self.insertVectorDB(document.doc_id)
        return BilanIndexation(document=document, caracteres_extraits=extraits,
                               securisation=rapport, chunks=len(chunks), vecteurs=vecteurs)

    async def reindexer(self, doc_id: str) -> BilanIndexation:
        """Rejoue ②③④ depuis la zone brute — après un changement de réglage."""
        document = await self._document(doc_id)
        extraits, rapport = await self.data_processing(doc_id)
        chunks = await self.data_chunking(doc_id)
        vecteurs = await self.insertVectorDB(doc_id)
        return BilanIndexation(document=document, caracteres_extraits=extraits,
                               securisation=rapport, chunks=len(chunks), vecteurs=vecteurs)

    async def supprimer(self, doc_id: str) -> int:
        """Retire un document de toutes les zones et de la base vectorielle.

        Indispensable au droit à l'effacement : un document supprimé du
        stockage mais resté dans la base continuerait d'être cité.
        """
        document = await self._document(doc_id)
        cles = [document.cle, self._cle_meta(doc_id), self._cle_processing(doc_id),
                self._cle_corpus(doc_id), self._cle_rapport(doc_id), self._cle_chunks(doc_id)]
        for cle in cles:
            await self.stockage.supprimer(cle)
        return await self.base.supprimer_document(doc_id)

    async def documents(self) -> list[DocumentBrut]:
        """Inventaire des documents déposés."""
        cles = await self.stockage.lister(f"{self.config.prefixe_brute}/")
        return [
            DocumentBrut.model_validate_json(await self.stockage.lire(c))
            for c in cles if c.endswith("/document.json")
        ]
