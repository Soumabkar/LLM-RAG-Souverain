"""Nettoyage et sécurisation du texte extrait.

Deux opérations distinctes, appliquées dans cet ordre :

1. **Nettoyage** — retire ce qui parasite la recherche sans porter de
   sens : caractères de contrôle, espaces insécables multiples, caractères
   invisibles.
2. **Sécurisation** — masque les données personnelles avant qu'elles
   n'entrent dans la base vectorielle. Un e-mail vectorisé ne s'efface
   plus : il faudrait réindexer tout le corpus.

« Supprimer les caractères spéciaux » ne veut pas dire tout ce qui n'est
pas alphanumérique : retirer les accents et la ponctuation dégraderait
le français et la qualité des embeddings. On ne retire que ce qui est
invisible ou non imprimable.
"""

from __future__ import annotations

import re
import unicodedata

from .config import RapportSecurisation

# ---------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------
# Espaces de largeur nulle, marques directionnelles, BOM : invisibles à
# l'écran mais présents dans le texte, ils cassent la correspondance
# lexicale et peuvent dissimuler du contenu.
_ESPACE_NULLE = re.compile(r"\u200b")
_INVISIBLES = re.compile(r"[\u200c-\u200f\u202a-\u202e\u2060-\u2064\ufeff\ufffd]")
_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ESPACES = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_LIGNES_VIDES = re.compile(r"\n{3,}")


def nettoyer(texte: str) -> tuple[str, int]:
    """Normalise le texte. Renvoie le texte et le nombre de caractères retirés."""
    avant = len(texte)
    # NFKC unifie les formes équivalentes : ligatures, chiffres pleine
    # chasse, espaces spéciales. « ﬁ » et « fi » deviennent identiques
    # pour la recherche.
    texte = unicodedata.normalize("NFKC", texte)
    # L'espace de largeur nulle marque une coupure de mot : la supprimer
    # purement et simplement collerait deux mots. On la remplace par un
    # espace ; les autres invisibles (liants, marques directionnelles)
    # sont retirés.
    texte = _ESPACE_NULLE.sub(" ", texte)
    texte = _INVISIBLES.sub("", texte)
    texte = _CONTROLE.sub("", texte)
    texte = texte.replace("\r\n", "\n").replace("\r", "\n")
    texte = _ESPACES.sub(" ", texte)
    texte = "\n".join(ligne.strip() for ligne in texte.split("\n"))
    texte = _LIGNES_VIDES.sub("\n\n", texte).strip()
    return texte, max(0, avant - len(texte))


# ---------------------------------------------------------------------
# Sécurisation
# ---------------------------------------------------------------------
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Numéros français : 0X XX XX XX XX, +33 X XX XX XX XX, avec ou sans
# séparateurs.
_TELEPHONE = re.compile(r"(?<!\d)(?:\+33\s?|0033\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)")

_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")

# Numéro de sécurité sociale : sexe, année, mois, département, commune,
# ordre, clé optionnelle.
_NIR = re.compile(
    r"(?<!\d)[12]\s?\d{2}\s?(?:0[1-9]|1[0-2]|[2-9]\d)\s?(?:\d{2}|2[AB])\s?\d{3}\s?\d{3}(?:\s?\d{2})?(?!\d)"
)

# Suites de 13 à 19 chiffres : candidates « carte bancaire ». Le contrôle
# de Luhn écarte ensuite les faux positifs — numéros de commande,
# identifiants techniques.
_CARTE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _luhn(numero: str) -> bool:
    chiffres = [int(c) for c in numero if c.isdigit()]
    total = 0
    for i, chiffre in enumerate(reversed(chiffres)):
        if i % 2 == 1:
            chiffre *= 2
            if chiffre > 9:
                chiffre -= 9
        total += chiffre
    return total % 10 == 0


def securiser(texte: str) -> tuple[str, RapportSecurisation]:
    """Masque les données personnelles.

    L'ordre compte : l'IBAN et le NIR passent avant la carte bancaire,
    dont le motif — une longue suite de chiffres — les engloberait.

    Le rapport compte les masquages sans jamais conserver les valeurs :
    il sert à l'audit, il ne doit pas devenir lui-même une fuite.
    """
    rapport = RapportSecurisation()

    texte, rapport.emails = _EMAIL.subn("[EMAIL]", texte)
    texte, rapport.ibans = _IBAN.subn("[IBAN]", texte)
    texte, rapport.nir = _NIR.subn("[NIR]", texte)

    def _carte(m: re.Match) -> str:
        if _luhn(m.group()):
            rapport.cartes += 1
            return "[CARTE]"
        return m.group()

    texte = _CARTE.sub(_carte, texte)
    texte, rapport.telephones = _TELEPHONE.subn("[TELEPHONE]", texte)

    return texte, rapport


def traiter(texte: str) -> tuple[str, RapportSecurisation]:
    """Nettoyage puis sécurisation."""
    propre, retires = nettoyer(texte)
    securise, rapport = securiser(propre)
    rapport.caracteres_retires = retires
    return securise, rapport
