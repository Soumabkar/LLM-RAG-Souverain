"""Administration du catalogue de modèles.

    python admin_catalogue.py list          # parc installé + catalogue
    python admin_catalogue.py check         # écarts entre les deux
    python admin_catalogue.py add llama3.2:latest "Llama 3.2" --desc "Généraliste"
    python admin_catalogue.py remove qwen2.5-coder:7b
    python admin_catalogue.py purge         # retire les modèles désinstallés
    python admin_catalogue.py grant DATA-01 llama3.2:latest

`add` refuse un modèle absent du serveur d'inférence : c'est le contrôle
de `create_model()`, jamais contourné ici. `purge` est le pendant pour
l'existant, quand le catalogue a été rempli sans vérification.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from Model.ai_model import AIModelError, installed_models
from Engine.db import close_pool
from Engine.models import model_llm, team


def _ligne(gauche: str, droite: str = "") -> None:
    print(f"  {gauche:<32} {droite}")


# =====================================================================
async def cmd_list() -> int:
    try:
        parc = sorted(await installed_models())
    except AIModelError as exc:
        print(f"✗ {exc}")
        return 1

    print(f"\nInstallés sur le serveur ({len(parc)})")
    for code in parc:
        _ligne(code)

    catalogue = await model_llm.catalogue(actifs_seulement=False)
    print(f"\nCatalogue ({len(catalogue)})")
    for m in catalogue:
        etat = "" if m["active"] else "  [désactivé]"
        presence = "✓" if m["code_model"] in parc else "✗ non installé"
        _ligne(f"{m['code_model']}{etat}", f"{m['display_name']:<20} {presence}")
    return 0


async def cmd_check() -> int:
    try:
        manquants = await model_llm.verifier_catalogue()
        absents = await model_llm.non_catalogues()
    except AIModelError as exc:
        print(f"✗ {exc}")
        return 1

    if manquants:
        print("\nCatalogués et actifs, mais PAS installés :")
        print("  (proposés aux utilisateurs, échoueront au premier message)")
        for code in manquants:
            _ligne(code, "→ ollama pull, ou purge")
    else:
        print("\n✓ Tous les modèles actifs du catalogue sont installés.")

    if absents:
        print("\nInstallés mais absents du catalogue :")
        for code in absents:
            _ligne(code, "→ add")

    return 1 if manquants else 0


async def cmd_add(code: str, nom: str, description: str | None) -> int:
    res = await model_llm(code, nom, description).create_model()
    print(("✓ " if res else "✗ ") + res.message)
    return 0 if res else 1


async def cmd_remove(code: str) -> int:
    res = await model_llm(code, "").delete_model()
    print(("✓ " if res else "✗ ") + res.message)
    return 0 if res else 1


async def cmd_purge(force: bool) -> int:
    try:
        manquants = await model_llm.verifier_catalogue()
    except AIModelError as exc:
        print(f"✗ {exc}")
        return 1

    if not manquants:
        print("✓ Rien à purger.")
        return 0

    print("Modèles catalogués mais non installés :")
    for code in manquants:
        _ligne(code)

    if not force:
        print("\nRelancer avec --force pour les retirer du catalogue.")
        print("Les habilitations des équipes partiront en cascade.")
        return 0

    for code in manquants:
        res = await model_llm(code, "").delete_model()
        print(("✓ " if res else "✗ ") + res.message)
    return 0


async def cmd_grant(code_team: str, code_model: str) -> int:
    res = await team(code_team, "").add_model(code_model)
    print(("✓ " if res else "✗ ") + res.message)
    return 0 if res else 1


# =====================================================================
async def main(argv: list[str] | None = None) -> int:
    parseur = argparse.ArgumentParser(prog="admin_catalogue", description=__doc__)
    sous = parseur.add_subparsers(dest="commande", required=True)

    sous.add_parser("list", help="parc installé et catalogue")
    sous.add_parser("check", help="écarts entre parc et catalogue")

    p_add = sous.add_parser("add", help="ajouter un modèle installé au catalogue")
    p_add.add_argument("code_model")
    p_add.add_argument("display_name")
    p_add.add_argument("--desc", default=None)

    p_rm = sous.add_parser("remove", help="retirer un modèle du catalogue")
    p_rm.add_argument("code_model")

    p_purge = sous.add_parser("purge", help="retirer les modèles désinstallés")
    p_purge.add_argument("--force", action="store_true")

    p_grant = sous.add_parser("grant", help="autoriser un modèle pour une équipe")
    p_grant.add_argument("code_team")
    p_grant.add_argument("code_model")

    args = parseur.parse_args(argv)

    try:
        if args.commande == "list":
            return await cmd_list()
        if args.commande == "check":
            return await cmd_check()
        if args.commande == "add":
            return await cmd_add(args.code_model, args.display_name, args.desc)
        if args.commande == "remove":
            return await cmd_remove(args.code_model)
        if args.commande == "purge":
            return await cmd_purge(args.force)
        if args.commande == "grant":
            return await cmd_grant(args.code_team, args.code_model)
        return 1
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
