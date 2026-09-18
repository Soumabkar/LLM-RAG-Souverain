"""Administration des comptes et des équipes.

    python admin_compte.py initialiser
    python admin_compte.py creer-equipe DATA-01 "Data Platform"
    python admin_compte.py creer-utilisateur ksoumahoro karim@entreprise.fr --equipe DATA-01
    python admin_compte.py creer-technique agent-rag agent-rag@interne "Agent RAG" --equipe AGENTS
    python admin_compte.py mot-de-passe karim@entreprise.fr
    python admin_compte.py lister

Complémentaire de `admin_catalogue.py`, qui gère les modèles.

À la différence de `demo_db_user_team.py`, ce script **ne vide jamais**
la base : la démonstration est jetable, pas les comptes.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys

from Engine.db import close_pool, get_connection
from Engine.models import model_llm, team, user
from Model.ai_model import AIModelError, installed_models

PREFIXES_EMBEDDING = ("nomic-embed", "mxbai-embed", "all-minilm", "bge-")


def ok(message: str) -> None:
    print(f"✓ {message}")


def ko(message: str) -> None:
    print(f"✗ {message}")


def resultat(res) -> int:
    (ok if res else ko)(res.message)
    return 0 if res else 1


# =====================================================================
async def cmd_creer_equipe(code: str, nom: str, email: str | None) -> int:
    return resultat(await team(code, nom, email).create_team())


# =====================================================================
async def cmd_creer_utilisateur(login: str, email: str, equipe: str | None) -> int:
    mot_de_passe = getpass.getpass("Mot de passe : ")
    if mot_de_passe != getpass.getpass("Confirmer    : "):
        ko("Les deux saisies diffèrent.")
        return 1
    if len(mot_de_passe) < 8:
        ko("Mot de passe trop court (8 caractères minimum).")
        return 1

    compte = user(login, mot_de_passe, email)
    code = resultat(await compte.create_user(equipe))
    if code == 0:
        modeles = [m["code_model"] for m in await compte.allowed_models()]
        print(f"  équipes : {compte.teams or 'aucune'}")
        print(f"  modèles : {modeles or 'aucun — habiliter son équipe'}")
    return code


# =====================================================================
async def cmd_creer_technique(login: str, email: str, description: str,
                              equipe: str | None) -> int:
    compte, retour = await user.creer_compte_technique(login, email, description, equipe)
    if compte is None:
        ko(retour)
        return 1

    ok(f"Compte technique '{email}' créé.")
    print(f"  mot de passe : {retour}")
    print("  À archiver dans KeePass : il n'est affiché qu'une fois.")
    print("  Ce compte ne peut pas ouvrir de session interactive.")
    return 0


# =====================================================================
async def cmd_mot_de_passe(email: str) -> int:
    compte = await user.load(email)
    if compte is None:
        ko(f"Compte '{email}' introuvable.")
        return 1
    if compte.technique:
        ko("Compte technique : il ne se connecte pas, changer son mot de passe est inutile.")
        return 1

    nouveau = getpass.getpass("Nouveau mot de passe : ")
    if nouveau != getpass.getpass("Confirmer            : "):
        ko("Les deux saisies diffèrent.")
        return 1

    compte.set_password(nouveau)
    return resultat(await compte.update_user())


# =====================================================================
async def cmd_lister() -> int:
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT u.email, u.login, u.technique,
                   coalesce(array_agg(t.code_team)
                            FILTER (WHERE t.code_team IS NOT NULL), '{}') AS equipes
              FROM llm_souverain.user_llm u
              LEFT JOIN llm_souverain.team_member t ON t.email = u.email
             GROUP BY u.email, u.login, u.technique
             ORDER BY u.technique, u.login
            """
        )
        comptes = await cur.fetchall()

    if not comptes:
        print("Aucun compte. Commencer par : python admin_compte.py initialiser")
        return 0

    print(f"\nComptes ({len(comptes)})")
    for c in comptes:
        marque = "[technique]" if c["technique"] else "           "
        print(f"  {marque} {c['login']:<14} {c['email']:<28} équipes={list(c['equipes'])}")

    equipes = await _equipes()
    print(f"\nÉquipes ({len(equipes)})")
    for e in equipes:
        print(f"  {e['code_team']:<10} {e['team_name']:<24} modèles={list(e['modeles'])}")
    return 0


async def _equipes() -> list[dict]:
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT t.code_team, t.team_name,
                   coalesce(array_agg(m.code_model)
                            FILTER (WHERE m.code_model IS NOT NULL), '{}') AS modeles
              FROM llm_souverain.team_llm t
              LEFT JOIN llm_souverain.team_model x ON x.code_team = t.code_team
              LEFT JOIN llm_souverain.model_llm  m ON m.code_model = x.code_model
                                                  AND m.active
             GROUP BY t.code_team, t.team_name
             ORDER BY t.code_team
            """
        )
        return await cur.fetchall()


# =====================================================================
async def cmd_initialiser() -> int:
    """Amorçage complet : catalogue, équipe, premier compte.

    Conçu pour une base neuve. Chaque étape est idempotente : relancer
    ne casse rien et complète ce qui manque.
    """
    print("Amorçage de la plateforme\n")

    # --- 1. Catalogue depuis le parc réellement installé
    try:
        parc = sorted(await installed_models())
    except AIModelError as exc:
        ko(f"Serveur d'inférence injoignable : {exc}")
        print("  Démarrer Ollama, puis relancer.")
        return 1

    conversation = [m for m in parc if not m.startswith(PREFIXES_EMBEDDING)]
    if not conversation:
        ko("Aucun modèle de conversation installé (ollama pull llama3.2).")
        return 1

    print(f"Parc détecté : {', '.join(conversation)}")
    for code in conversation:
        nom = code.split(":")[0].replace("-", " ").title()[:50]
        res = await model_llm(code, nom, f"Modèle {nom}").create_model()
        print(f"  {'✓' if res else '·'} {res.message}")

    # --- 2. Équipe et habilitations
    equipe = input("\nCode de la première équipe [DATA-01] : ").strip() or "DATA-01"
    nom_equipe = input(f"Nom de l'équipe [{equipe}] : ").strip() or equipe
    res = await team(equipe, nom_equipe).create_team()
    print(f"  {'✓' if res else '·'} {res.message}")

    for code in conversation:
        res = await team(equipe, "").add_model(code)
        print(f"  {'✓' if res else '·'} {res.message}")

    # --- 3. Premier compte
    print()
    login = input("Login du premier utilisateur [ksoumahoro] : ").strip() or "ksoumahoro"
    email = input("Email [karim@entreprise.fr] : ").strip() or "karim@entreprise.fr"

    if await user.find(email) is not None:
        print(f"· Le compte '{email}' existe déjà.")
    else:
        code_retour = await cmd_creer_utilisateur(login, email, equipe)
        if code_retour != 0:
            return code_retour

    print("\nPrêt. Lancer l'interface :")
    print("  python -m chainlit run Interface/chainlit_app.py -w")
    return 0


# =====================================================================
async def main(argv: list[str] | None = None) -> int:
    parseur = argparse.ArgumentParser(prog="admin_compte", description=__doc__)
    sous = parseur.add_subparsers(dest="commande", required=True)

    sous.add_parser("initialiser", help="amorçage complet d'une base neuve")
    sous.add_parser("lister", help="comptes et équipes")

    p = sous.add_parser("creer-equipe")
    p.add_argument("code")
    p.add_argument("nom")
    p.add_argument("--email", default=None)

    p = sous.add_parser("creer-utilisateur")
    p.add_argument("login")
    p.add_argument("email")
    p.add_argument("--equipe", default=None)

    p = sous.add_parser("creer-technique")
    p.add_argument("login")
    p.add_argument("email")
    p.add_argument("description")
    p.add_argument("--equipe", default=None)

    p = sous.add_parser("mot-de-passe")
    p.add_argument("email")

    args = parseur.parse_args(argv)

    try:
        if args.commande == "initialiser":
            return await cmd_initialiser()
        if args.commande == "lister":
            return await cmd_lister()
        if args.commande == "creer-equipe":
            return await cmd_creer_equipe(args.code, args.nom, args.email)
        if args.commande == "creer-utilisateur":
            return await cmd_creer_utilisateur(args.login, args.email, args.equipe)
        if args.commande == "creer-technique":
            return await cmd_creer_technique(
                args.login, args.email, args.description, args.equipe
            )
        if args.commande == "mot-de-passe":
            return await cmd_mot_de_passe(args.email)
        return 1
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
