"""Extraction du texte selon le format source.

Chaque extracteur renvoie du texte brut, avant nettoyage : c'est ce qui
est écrit en zone `data-processing`, pour pouvoir diagnostiquer une
extraction défaillante indépendamment du nettoyage.

Les bibliothèques des formats binaires (pypdf, pyarrow, fastavro) sont
importées à la demande : un déploiement qui n'ingère que du HTML et du
Markdown n'a pas à les installer.
"""

from __future__ import annotations

import csv
import io
import json
from html.parser import HTMLParser

from .config import TypeDocument


class ExtractionError(ValueError):
    """Contenu illisible dans le format annoncé."""


# =====================================================================
class _ExtracteurHTML(HTMLParser):
    """Texte visible d'une page HTML.

    Les balises script, style, noscript et template sont ignorées avec
    leur contenu. C'est aussi une précaution de sécurité : du texte
    invisible pour un lecteur humain peut porter des instructions
    destinées au modèle — c'est le principe de l'injection de prompt
    indirecte.
    """

    IGNOREES = {"script", "style", "noscript", "template", "head", "svg"}
    BLOCS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "header", "footer", "table", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._morceaux: list[str] = []
        self._profondeur_ignoree = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNOREES:
            self._profondeur_ignoree += 1
        elif tag in self.BLOCS:
            self._morceaux.append("\n")

    def handle_endtag(self, tag):
        if tag in self.IGNOREES and self._profondeur_ignoree:
            self._profondeur_ignoree -= 1
        elif tag in self.BLOCS:
            self._morceaux.append("\n")

    def handle_data(self, data):
        if not self._profondeur_ignoree:
            self._morceaux.append(data)

    def texte(self) -> str:
        return "".join(self._morceaux)


def _html(contenu: bytes) -> str:
    extracteur = _ExtracteurHTML()
    extracteur.feed(contenu.decode("utf-8", errors="replace"))
    return extracteur.texte()


def _markdown(contenu: bytes) -> str:
    # Le Markdown est déjà du texte structuré : on le garde tel quel. Les
    # titres et listes aident le découpage à respecter la structure.
    return contenu.decode("utf-8", errors="replace")


def _csv(contenu: bytes) -> str:
    """Une ligne par enregistrement, sous la forme « colonne : valeur ».

    Répéter le nom de colonne sur chaque ligne alourdit le texte, mais
    rend chaque chunk compréhensible isolément : un extrait « 42, Paris,
    actif » ne veut rien dire sans son en-tête.
    """
    texte = contenu.decode("utf-8-sig", errors="replace")
    try:
        dialecte = csv.Sniffer().sniff(texte[:4096], delimiters=",;\t|")
    except csv.Error:
        dialecte = csv.excel
    lecteur = csv.DictReader(io.StringIO(texte), dialect=dialecte)
    lignes = []
    for enregistrement in lecteur:
        champs = [f"{k.strip()} : {str(v).strip()}" for k, v in enregistrement.items()
                  if k and v not in (None, "")]
        if champs:
            lignes.append(" | ".join(champs))
    return "\n".join(lignes)


def _pdf(contenu: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError("pypdf requis pour les PDF : pip install pypdf") from exc
    try:
        lecteur = PdfReader(io.BytesIO(contenu))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"PDF illisible : {exc}") from exc
    pages = [(page.extract_text() or "") for page in lecteur.pages]
    texte = "\n\n".join(p for p in pages if p.strip())
    if not texte.strip():
        # Un PDF sans texte extractible est presque toujours un scan :
        # il faut de l'OCR, pas un meilleur extracteur.
        raise ExtractionError(
            "Aucun texte extractible — PDF probablement scanné, OCR nécessaire."
        )
    return texte


def _enregistrements(lignes: list[dict]) -> str:
    return "\n".join(
        " | ".join(f"{k} : {v}" for k, v in ligne.items() if v not in (None, ""))
        for ligne in lignes
    )


def _parquet(contenu: bytes) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ExtractionError("pyarrow requis pour Parquet : pip install pyarrow") from exc
    table = pq.read_table(io.BytesIO(contenu))
    return _enregistrements(table.to_pylist())


def _avro(contenu: bytes) -> str:
    try:
        import fastavro
    except ImportError as exc:
        raise ExtractionError("fastavro requis pour Avro : pip install fastavro") from exc
    lecteur = fastavro.reader(io.BytesIO(contenu))
    lignes = [
        {k: (json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v)
         for k, v in enregistrement.items()}
        for enregistrement in lecteur
    ]
    return _enregistrements(lignes)


EXTRACTEURS = {
    TypeDocument.HTML: _html,
    TypeDocument.MARKDOWN: _markdown,
    TypeDocument.CSV: _csv,
    TypeDocument.PDF: _pdf,
    TypeDocument.PARQUET: _parquet,
    TypeDocument.AVRO: _avro,
}


def extraire_texte(contenu: bytes, type_doc: TypeDocument) -> str:
    """Texte brut d'un document, selon son format."""
    try:
        return EXTRACTEURS[type_doc](contenu)
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Extraction {type_doc.value} impossible : {exc}") from exc
