"""Hachage des mots de passe.

Format stocké : ``bcrypt-sha256$<hash bcrypt de 60 caractères>`` (74 au total,
la colonne en accepte 256).

Le pré-hachage SHA-256 encodé en base64 (44 octets constants) contourne la
limite de 72 octets de bcrypt. Sans lui, `bcrypt >= 4.1` lève une `ValueError`
sur une phrase de passe un peu longue — 40 caractères accentués suffisent,
puisqu'ils comptent double en UTF-8 — et les versions plus anciennes
tronquaient silencieusement, ce qui est pire : deux mots de passe partageant
leurs 72 premiers octets se validaient l'un pour l'autre.

Les hash bcrypt nus ($2a$/$2b$/$2y$) produits par la version précédente
restent vérifiables ; `needs_rehash` permet de les migrer à la connexion.
"""

from __future__ import annotations

import base64
import hashlib
import re

import bcrypt

PREFIX = "bcrypt-sha256$"

# Un hash bcrypt fait exactement 60 caractères : $2b$ + coût sur 2 + $ + 53.
_BCRYPT_RE = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")


def _digest(plain: str) -> bytes:
    """SHA-256 du mot de passe, encodé en base64 : toujours 44 octets."""
    return base64.b64encode(hashlib.sha256(plain.encode("utf-8")).digest())


def hash_password(plain: str) -> str:
    """Hache un mot de passe de longueur quelconque."""
    if not plain:
        raise ValueError("Le mot de passe ne peut pas être vide.")
    return PREFIX + bcrypt.hashpw(_digest(plain), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Vérifie un mot de passe contre un hash, quel que soit son format."""
    if not plain or not hashed:
        return False
    try:
        if hashed.startswith(PREFIX):
            return bcrypt.checkpw(_digest(plain), hashed[len(PREFIX) :].encode("utf-8"))
        if _BCRYPT_RE.match(hashed):
            # Hash historique : bcrypt nu, tronqué à 72 octets comme à l'origine.
            return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
        return False
    except (ValueError, TypeError):
        return False


def is_hashed(value: str) -> bool:
    """Vrai si la valeur est déjà un hash.

    Le format complet est vérifié : un simple test de préfixe classerait
    « $2b$mon_mot_de_passe » comme un hash et le stockerait en clair.
    """
    if not value:
        return False
    if value.startswith(PREFIX):
        return bool(_BCRYPT_RE.match(value[len(PREFIX) :]))
    return bool(_BCRYPT_RE.match(value))


def needs_rehash(hashed: str) -> bool:
    """Vrai pour un hash au format historique, à remplacer à la connexion."""
    return is_hashed(hashed) and not hashed.startswith(PREFIX)


def ensure_hashed(value: str) -> str:
    """Hache la valeur si ce n'en est pas déjà un hash (idempotent)."""
    return value if is_hashed(value) else hash_password(value)
