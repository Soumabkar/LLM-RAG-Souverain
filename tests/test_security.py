"""Tests du hachage des mots de passe. Aucun accès base."""

from __future__ import annotations

import bcrypt
import pytest

from Engine.security import (
    PREFIX,
    ensure_hashed,
    hash_password,
    is_hashed,
    needs_rehash,
    verify_password,
)


class TestHashPassword:
    def test_le_hash_ne_contient_pas_le_mot_de_passe(self):
        assert "MotDePasse" not in hash_password("MotDePasse!2026")

    def test_deux_hash_du_meme_mot_de_passe_different(self):
        """Le sel rend chaque hash unique : deux comptes avec le même mot
        de passe ne sont pas repérables en base."""
        assert hash_password("identique") != hash_password("identique")

    def test_longueur_compatible_avec_la_colonne(self):
        assert len(hash_password("x")) <= 256

    def test_mot_de_passe_vide_refuse(self):
        with pytest.raises(ValueError):
            hash_password("")


class TestLimite72Octets:
    """bcrypt lève une ValueError au-delà de 72 octets. Le pré-hachage
    SHA-256 doit rendre la longueur indifférente."""

    @pytest.mark.parametrize(
        "mot_de_passe",
        [
            "A" * 100,
            "é" * 60,  # 120 octets en UTF-8
            "phrase de passe très longue " * 10,
            "🔐" * 30,  # 120 octets
        ],
        ids=["100 ascii", "60 accents", "phrase longue", "emoji"],
    )
    def test_mot_de_passe_long_accepte(self, mot_de_passe):
        assert verify_password(mot_de_passe, hash_password(mot_de_passe))

    def test_les_73_premiers_octets_ne_suffisent_pas(self):
        """Sans pré-hachage, bcrypt ignorait tout au-delà de 72 octets :
        deux mots de passe identiques sur 72 octets se validaient l'un
        pour l'autre."""
        h = hash_password("A" * 100)
        assert not verify_password("A" * 72, h)


class TestVerifyPassword:
    def test_mot_de_passe_correct(self):
        assert verify_password("secret", hash_password("secret"))

    def test_mot_de_passe_incorrect(self):
        assert not verify_password("Secret", hash_password("secret"))

    @pytest.mark.parametrize("hachage", ["", "pas-un-hash", "$2b$tronque", "null"])
    def test_hash_invalide_renvoie_false_sans_lever(self, hachage):
        assert verify_password("secret", hachage) is False

    def test_mot_de_passe_vide(self):
        assert not verify_password("", hash_password("secret"))


class TestCompatibiliteAscendante:
    """Les hash bcrypt nus créés par la version précédente doivent rester
    vérifiables, sinon tous les comptes existants sont verrouillés."""

    def test_hash_bcrypt_historique_accepte(self):
        legacy = bcrypt.hashpw(b"ancien", bcrypt.gensalt()).decode()
        assert verify_password("ancien", legacy)

    def test_hash_historique_signale_pour_rehachage(self):
        legacy = bcrypt.hashpw(b"ancien", bcrypt.gensalt()).decode()
        assert needs_rehash(legacy)
        assert not needs_rehash(hash_password("nouveau"))


class TestIsHashed:
    def test_reconnait_le_format_courant(self):
        assert is_hashed(hash_password("x"))

    def test_reconnait_le_format_historique(self):
        assert is_hashed(bcrypt.hashpw(b"x", bcrypt.gensalt()).decode())

    @pytest.mark.parametrize(
        "valeur",
        ["motdepasse", "$2b$monMotDePasse", "$2b$12$trop-court", PREFIX],
        ids=["texte simple", "faux positif", "tronque", "prefixe seul"],
    )
    def test_ne_confond_pas_un_mot_de_passe_avec_un_hash(self, valeur):
        assert not is_hashed(valeur)


class TestEnsureHashed:
    def test_hache_un_mot_de_passe_en_clair(self):
        assert is_hashed(ensure_hashed("motdepasse"))

    def test_idempotent_sur_un_hash(self):
        """Recharger un utilisateur depuis la base puis le sauvegarder ne
        doit pas hacher le hash une deuxième fois."""
        h = hash_password("secret")
        assert ensure_hashed(h) == h
        assert verify_password("secret", ensure_hashed(h))
