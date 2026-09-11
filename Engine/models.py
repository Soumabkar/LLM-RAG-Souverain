"""Classes métier `user` et `team` adossées au schéma llm_souverain.

Tables ciblées :
    llm_souverain.team_llm(code_team PK, team_name, email)
    llm_souverain.user_llm(login UNIQUE, password, email PK)
    llm_souverain.team_member(code_team, email)  -- association N-N

Un utilisateur peut appartenir à plusieurs équipes. L'appartenance est
portée exclusivement par team_member : `user.teams` est une liste de
codes équipe, `team.members` une liste d'emails.

Chaque méthode renvoie un objet `Result` (booléen + message) plutôt que de
lever une exception : c'est la « notification » décrite dans les spécifications.
`Result` est utilisable directement dans un `if`.

Toutes les méthodes qui touchent la base sont des coroutines : elles
s'appellent avec `await`. Les helpers purement mémoire (`set_password`,
`code_team`) restent synchrones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg

from .db import get_connection
from .security import ensure_hashed, needs_rehash, verify_password

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    ok: bool
    message: str
    data: dict[str, Any] | None = field(default=None)

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------
# Validation des entrées
#
# Les longueurs reprennent celles des colonnes. Sans ce contrôle en amont,
# PostgreSQL lève une StringDataRightTruncation dont le message
# (« value too long for type character varying(50) ») ne dit pas quel champ
# est en cause : inexploitable dans une IHM.
# ---------------------------------------------------------------------
MAX_LEN = {
    "login": 50,
    "email": 256,
    "password": 256,
    "code_team": 256,
    "team_name": 50,
}


def _check(champ: str, valeur: Any, libelle: str, obligatoire: bool = True) -> str | None:
    """Renvoie un message d'erreur, ou None si la valeur est acceptable."""
    if valeur is None or (isinstance(valeur, str) and not valeur.strip()):
        return f"Le champ {libelle} est obligatoire." if obligatoire else None
    if not isinstance(valeur, str):
        return f"Le champ {libelle} doit être une chaîne de caractères."
    limite = MAX_LEN[champ]
    if len(valeur) > limite:
        return f"Le champ {libelle} fait {len(valeur)} caractères (maximum {limite})."
    return None


def _first_error(*controles: str | None) -> str | None:
    for message in controles:
        if message is not None:
            return message
    return None


# =====================================================================
# user
# =====================================================================
class user:
    """Compte utilisateur. La clé primaire fonctionnelle est l'email."""

    def __init__(self, login: str, password: str, email: str):
        self.login = login
        self.password = password  # en clair en mémoire, haché à l'écriture
        self.email = email
        self.teams: list[str] = []

    def __repr__(self) -> str:  # pragma: no cover
        return f"user(login={self.login!r}, email={self.email!r}, teams={self.teams!r})"

    @property
    def code_team(self) -> str | None:
        """Compatibilité avec l'ancien modèle : première équipe, ou None."""
        return self.teams[0] if self.teams else None

    # -----------------------------------------------------------------
    async def create_user(self, code_team: str | None = None) -> Result:
        """Crée l'utilisateur, et le rattache à `code_team` si fourni.

        Les deux écritures sont dans la même transaction : si l'équipe
        n'existe pas, l'utilisateur n'est pas créé non plus.
        """
        erreur = _first_error(
            _check("login", self.login, "login"),
            _check("email", self.email, "email"),
            _check("password", self.password, "password"),
            _check("code_team", code_team, "code_team", obligatoire=False),
        )
        if erreur:
            return Result(False, erreur)

        try:
            async with get_connection() as conn, conn.cursor() as cur:
                if code_team is not None:
                    await cur.execute(
                        "SELECT 1 FROM llm_souverain.team_llm WHERE code_team = %s",
                        (code_team,),
                    )
                    if await cur.fetchone() is None:
                        return Result(False, f"Le code équipe '{code_team}' n'existe pas.")

                await cur.execute(
                    """
                    INSERT INTO llm_souverain.user_llm (login, password, email)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email) DO NOTHING
                    """,
                    (self.login, ensure_hashed(self.password), self.email),
                )
                if cur.rowcount == 0:
                    return Result(False, f"L'utilisateur '{self.email}' existe déjà.")

                if code_team is not None:
                    await cur.execute(
                        """
                        INSERT INTO llm_souverain.team_member (code_team, email)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (code_team, self.email),
                    )

            self.teams = [code_team] if code_team else []
            logger.info("Utilisateur %s créé (équipes : %s)", self.email, self.teams)
            return Result(True, f"Utilisateur '{self.email}' créé.")

        except psycopg.errors.UniqueViolation:
            return Result(False, f"Le login '{self.login}' est déjà utilisé.")
        except psycopg.Error as exc:
            logger.exception("Échec de création de l'utilisateur %s", self.email)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def delete_user(self) -> Result:
        """Supprime l'utilisateur. Ses adhésions partent en cascade."""
        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM llm_souverain.user_llm WHERE email = %s",
                    (self.email,),
                )
                if cur.rowcount == 0:
                    return Result(False, f"L'utilisateur '{self.email}' n'existe pas.")

            self.teams = []
            logger.info("Utilisateur %s supprimé", self.email)
            return Result(True, f"Utilisateur '{self.email}' supprimé.")

        except psycopg.Error as exc:
            logger.exception("Échec de suppression de l'utilisateur %s", self.email)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def update_user(self) -> Result:
        """Met à jour login et password.

        L'email sert d'identifiant de recherche (clé primaire) : voir
        `change_email`. Les équipes se gèrent via `join_team` / `leave_team`.
        """
        erreur = _first_error(
            _check("login", self.login, "login"),
            _check("password", self.password, "password"),
        )
        if erreur:
            return Result(False, erreur)

        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE llm_souverain.user_llm
                       SET login    = %s,
                           password = %s
                     WHERE email = %s
                    """,
                    (self.login, ensure_hashed(self.password), self.email),
                )
                if cur.rowcount == 0:
                    return Result(False, f"L'utilisateur '{self.email}' n'existe pas.")

            logger.info("Utilisateur %s mis à jour", self.email)
            return Result(True, f"Utilisateur '{self.email}' mis à jour.")

        except psycopg.errors.UniqueViolation:
            return Result(False, f"Le login '{self.login}' est déjà utilisé.")
        except psycopg.Error as exc:
            logger.exception("Échec de mise à jour de l'utilisateur %s", self.email)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def change_email(self, new_email: str) -> Result:
        """Change la clé primaire. Les adhésions suivent (ON UPDATE CASCADE)."""
        erreur = _check("email", new_email, "email")
        if erreur:
            return Result(False, erreur)

        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "UPDATE llm_souverain.user_llm SET email = %s WHERE email = %s",
                    (new_email, self.email),
                )
                if cur.rowcount == 0:
                    return Result(False, f"L'utilisateur '{self.email}' n'existe pas.")

            self.email = new_email
            return Result(True, f"Email mis à jour : '{new_email}'.")

        except psycopg.errors.UniqueViolation:
            return Result(False, f"L'email '{new_email}' est déjà utilisé.")
        except psycopg.Error as exc:
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def join_team(self, code_team: str) -> Result:
        """Rejoint une équipe (symétrique de team.add_member_team)."""
        return await team(code_team, "").add_member_team(self)

    async def leave_team(self, code_team: str) -> Result:
        """Quitte une équipe (symétrique de team.delete_member_team)."""
        return await team(code_team, "").delete_member_team(self)

    async def load_teams(self) -> list[str]:
        """Charge les codes équipe de l'utilisateur dans `self.teams`."""
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT code_team
                  FROM llm_souverain.team_member
                 WHERE email = %s
                 ORDER BY code_team
                """,
                (self.email,),
            )
            rows = await cur.fetchall()
        self.teams = [row["code_team"] for row in rows]
        return self.teams

    async def is_member_of(self, code_team: str) -> bool:
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT 1 FROM llm_souverain.team_member
                 WHERE email = %s AND code_team = %s
                """,
                (self.email, code_team),
            )
            return await cur.fetchone() is not None

    # -----------------------------------------------------------------
    def set_password(self, plain_password: str) -> None:
        """Change le mot de passe en mémoire ; appeler `update_user` ensuite."""
        self.password = plain_password

    async def check_password(self, plain_password: str) -> bool:
        """Vérifie un mot de passe contre le hash stocké en base."""
        row = await self.find(self.email)
        return bool(row) and verify_password(plain_password, row["password"])

    # -----------------------------------------------------------------
    @staticmethod
    async def find(email: str) -> dict[str, Any] | None:
        """Renvoie la ligne brute de user_llm, ou None."""
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT login, password, email
                  FROM llm_souverain.user_llm
                 WHERE email = %s
                """,
                (email,),
            )
            return await cur.fetchone()

    @staticmethod
    async def find_by_login(login: str) -> dict[str, Any] | None:
        """Renvoie la ligne brute de user_llm à partir du login, ou None."""
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT login, password, email
                  FROM llm_souverain.user_llm
                 WHERE login = %s
                """,
                (login,),
            )
            return await cur.fetchone()

    @classmethod
    async def load(cls, email: str) -> "user | None":
        """Instancie un `user` à partir de la base (password = hash)."""
        row = await cls.find(email)
        if row is None:
            return None
        instance = cls(row["login"], row["password"], row["email"])
        await instance.load_teams()
        return instance


# =====================================================================
# team
# =====================================================================
class team:
    """Équipe. Les membres vivent dans llm_souverain.team_member."""

    def __init__(self, code_team: str, team_name: str, email: str | None = None):
        self.code_team = code_team
        self.team_name = team_name
        self.email = email
        self.members: list[str] = []

    def __repr__(self) -> str:  # pragma: no cover
        return f"team(code_team={self.code_team!r}, team_name={self.team_name!r})"

    # -----------------------------------------------------------------
    async def create_team(self) -> Result:
        """Insère l'équipe dans team_llm."""
        erreur = _first_error(
            _check("code_team", self.code_team, "code_team"),
            _check("team_name", self.team_name, "nom d'équipe"),
            _check("email", self.email, "email", obligatoire=False),
        )
        if erreur:
            return Result(False, erreur)

        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO llm_souverain.team_llm (code_team, team_name, email)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (code_team) DO NOTHING
                    """,
                    (self.code_team, self.team_name, self.email),
                )
                if cur.rowcount == 0:
                    return Result(False, f"L'équipe '{self.code_team}' existe déjà.")

            logger.info("Équipe %s créée", self.code_team)
            return Result(True, f"Équipe '{self.code_team}' créée.")

        except psycopg.Error as exc:
            logger.exception("Échec de création de l'équipe %s", self.code_team)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def update_team(self) -> Result:
        """Met à jour le nom, et l'email s'il est renseigné.

        `self.email is None` laisse la colonne inchangée : sans cela,
        `team("DATA-01", "Nouveau nom").update_team()` effaçait l'adresse
        de contact sans que l'appelant l'ait demandé. Pour la vider,
        passer une chaîne vide.
        """
        erreur = _first_error(
            _check("team_name", self.team_name, "nom d'équipe"),
            _check("email", self.email, "email", obligatoire=False),
        )
        if erreur:
            return Result(False, erreur)

        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE llm_souverain.team_llm
                       SET team_name = %s,
                           email     = COALESCE(%s, email)
                     WHERE code_team = %s
                    """,
                    (self.team_name, self.email, self.code_team),
                )
                if cur.rowcount == 0:
                    return Result(False, f"L'équipe '{self.code_team}' n'existe pas.")

            logger.info("Équipe %s mise à jour", self.code_team)
            return Result(True, f"Équipe '{self.code_team}' mise à jour.")

        except psycopg.Error as exc:
            logger.exception("Échec de mise à jour de l'équipe %s", self.code_team)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def delete_team(self) -> Result:
        """Supprime l'équipe.

        Les adhésions sont supprimées en cascade ; les comptes utilisateurs
        eux-mêmes ne sont pas touchés.
        """
        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) AS nb FROM llm_souverain.team_member WHERE code_team = %s",
                    (self.code_team,),
                )
                nb_members = (await cur.fetchone())["nb"]

                await cur.execute(
                    "DELETE FROM llm_souverain.team_llm WHERE code_team = %s",
                    (self.code_team,),
                )
                if cur.rowcount == 0:
                    return Result(False, f"L'équipe '{self.code_team}' n'existe pas.")

            self.members = []
            logger.info("Équipe %s supprimée (%s adhésions)", self.code_team, nb_members)
            return Result(
                True,
                f"Équipe '{self.code_team}' supprimée, {nb_members} adhésion(s) retirée(s).",
            )

        except psycopg.Error as exc:
            logger.exception("Échec de suppression de l'équipe %s", self.code_team)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def add_member_team(self, user: user) -> Result:
        """Ajoute une adhésion (code_team, email) dans team_member."""
        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO llm_souverain.team_member (code_team, email)
                    VALUES (%s, %s)
                    ON CONFLICT (code_team, email) DO NOTHING
                    """,
                    (self.code_team, user.email),
                )
                if cur.rowcount == 0:
                    return Result(
                        False,
                        f"'{user.email}' est déjà membre de l'équipe '{self.code_team}'.",
                    )

            await user.load_teams()
            await self.load_members()
            logger.info("Utilisateur %s ajouté à l'équipe %s", user.email, self.code_team)
            return Result(True, f"'{user.email}' ajouté à l'équipe '{self.code_team}'.")

        except psycopg.errors.ForeignKeyViolation as exc:
            # exc.diag.constraint_name est renseigné par PostgreSQL : plus
            # fiable qu'une recherche de sous-chaîne dans le message, qui
            # dépend de la locale du serveur.
            if exc.diag.constraint_name == "fk_team_member_user":
                return Result(False, f"L'utilisateur '{user.email}' n'existe pas.")
            return Result(False, f"L'équipe '{self.code_team}' n'existe pas.")
        except psycopg.Error as exc:
            logger.exception("Échec d'ajout de %s à l'équipe %s", user.email, self.code_team)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def delete_member_team(self, user: user) -> Result:
        """Retire l'adhésion. Le compte utilisateur n'est pas supprimé."""
        try:
            async with get_connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM llm_souverain.team_member
                     WHERE code_team = %s AND email = %s
                    """,
                    (self.code_team, user.email),
                )
                if cur.rowcount == 0:
                    return Result(
                        False,
                        f"'{user.email}' n'est pas membre de l'équipe '{self.code_team}'.",
                    )

            await user.load_teams()
            await self.load_members()
            logger.info("Utilisateur %s retiré de l'équipe %s", user.email, self.code_team)
            return Result(True, f"'{user.email}' retiré de l'équipe '{self.code_team}'.")

        except psycopg.Error as exc:
            logger.exception("Échec de retrait de %s de l'équipe %s", user.email, self.code_team)
            return Result(False, f"Erreur base de données : {exc}")

    # -----------------------------------------------------------------
    async def load_members(self) -> list[str]:
        """Charge la liste des emails des membres dans `self.members`."""
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT email
                  FROM llm_souverain.team_member
                 WHERE code_team = %s
                 ORDER BY email
                """,
                (self.code_team,),
            )
            rows = await cur.fetchall()
        self.members = [row["email"] for row in rows]
        return self.members

    @classmethod
    async def load(cls, code_team: str) -> "team | None":
        """Instancie une `team` à partir de la base."""
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT code_team, team_name, email
                  FROM llm_souverain.team_llm
                 WHERE code_team = %s
                """,
                (code_team,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        instance = cls(row["code_team"], row["team_name"], row["email"])
        await instance.load_members()
        return instance


# =====================================================================
# Authentification
# =====================================================================
# Hash d'une valeur arbitraire, comparé quand le compte n'existe pas : sans
# lui, une réponse instantanée sur un identifiant inconnu contre ~300 ms sur
# un identifiant valide permettrait d'énumérer les comptes au chronomètre.
_HASH_FACTICE = "bcrypt-sha256$$2b$12$aDEIR62FPOGTke1QhWYU7ufdqSoCnX2TXoWgXJmm1wYoDM6WC2RUq"


async def authenticate(identifiant: str, mot_de_passe: str) -> user | None:
    """Vérifie un couple identifiant/mot de passe.

    `identifiant` accepte indifféremment l'email ou le login : les deux sont
    uniques en base.

    Renvoie le compte chargé (équipes comprises) en cas de succès, None
    sinon — sans distinguer « compte inconnu » de « mot de passe faux »,
    pour ne pas révéler quels comptes existent.
    """
    if not identifiant or not mot_de_passe:
        return None

    identifiant = identifiant.strip()
    ligne = await user.find(identifiant)
    if ligne is None:
        ligne = await user.find_by_login(identifiant)

    if ligne is None:
        verify_password(mot_de_passe, _HASH_FACTICE)  # temps constant
        logger.info("Authentification refusée : identifiant '%s' inconnu", identifiant)
        return None

    if not verify_password(mot_de_passe, ligne["password"]):
        logger.info("Authentification refusée : mot de passe invalide pour %s", ligne["email"])
        return None

    compte = user(ligne["login"], ligne["password"], ligne["email"])
    await compte.load_teams()

    # Migration opportuniste : le seul moment où le mot de passe en clair est
    # disponible pour re-hacher un format historique.
    if needs_rehash(ligne["password"]):
        compte.set_password(mot_de_passe)
        await compte.update_user()
        logger.info("Hash historique migré pour %s", compte.email)

    logger.info("Authentification réussie : %s (%s)", compte.email, ",".join(compte.teams) or "-")
    return compte
