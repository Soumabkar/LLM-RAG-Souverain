"""Démonstration du socle : python demo_db_user_team.py

Utilisateurs, équipes, catalogue de modèles, habilitations et comptes
techniques — le tout contre la base réelle.

Le catalogue est construit à partir des modèles **réellement installés**
sur le serveur d'inférence : la démonstration reflète donc ton parc, pas
une liste figée. Si Ollama est injoignable, un parc fictif prend le
relais et le contrôle d'installation est désactivé.

Le script est rejouable : il vide les tables au début comme à la fin.
"""

from __future__ import annotations

import asyncio
import logging

from Engine.db import close_pool, get_connection
from Engine.models import model_llm, team, user
from Model.ai_model import AIModelError, installed_models

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logging.getLogger("Engine.models").setLevel(logging.WARNING)
for bruyant in ("httpx", "httpx2", "openai", "openai._base_client", "Model.ai_model"):
    logging.getLogger(bruyant).setLevel(logging.WARNING)

TABLES = (
    "llm_souverain.team_model, llm_souverain.model_llm, "
    "llm_souverain.team_member, llm_souverain.user_llm, llm_souverain.team_llm"
)

# Les modèles d'embeddings servent à l'indexation du RAG, pas à la
# conversation : ils n'ont rien à faire dans un sélecteur de chat.
PREFIXES_EMBEDDING = ("nomic-embed", "mxbai-embed", "all-minilm", "bge-")

PARC_FICTIF = ["llama3.2:latest", "qwen2.5:latest", "mistral:7b"]


def titre(texte: str) -> None:
    print(f"\n{'─' * 72}\n  {texte}\n{'─' * 72}")


async def vider() -> None:
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(f"TRUNCATE {TABLES} CASCADE")


# =====================================================================
async def detecter_parc() -> tuple[list[str], bool]:
    """Modèles de conversation installés, et présence réelle du serveur."""
    try:
        parc = sorted(await installed_models())
    except AIModelError as exc:
        print(f"Serveur d'inférence injoignable ({str(exc)[:60]}…)")
        print(f"Parc fictif utilisé : {', '.join(PARC_FICTIF)}")
        return PARC_FICTIF, False

    conversation = [m for m in parc if not m.startswith(PREFIXES_EMBEDDING)]
    embeddings = [m for m in parc if m.startswith(PREFIXES_EMBEDDING)]

    print(f"Parc installé : {', '.join(conversation) or 'aucun modèle de conversation'}")
    if embeddings:
        print(f"Écartés (embeddings, réservés au RAG) : {', '.join(embeddings)}")

    if not conversation:
        print("Aucun modèle de conversation — bascule sur un parc fictif.")
        return PARC_FICTIF, False

    return conversation, True


# =====================================================================
async def catalogue(parc: list[str], reel: bool) -> None:
    """Alimente le catalogue et montre le refus d'un modèle absent."""
    titre("1. Catalogue — un modèle n'entre que s'il est installé")

    for code in parc:
        nom = code.split(":")[0].replace("-", " ").title()
        resultat = await model_llm(code, nom[:50], f"Modèle {nom}").create_model(verify=reel)
        print(f"  {'✓' if resultat else '✗'} {resultat.message}")

    # Contrôle actif sur un modèle qui n'existe nulle part. Sans serveur,
    # on injecte un vérificateur simulé : le message attendu est « pas
    # installé », pas « serveur injoignable » — deux causes distinctes
    # qui appellent deux actions différentes.
    async def parc_simule(code: str) -> bool:
        return code in parc

    absent = await model_llm("gpt-oss:120b", "GPT-OSS").create_model(
        verify=True, verifier=None if reel else parc_simule
    )
    print(f"  ✗ {absent.message}")


# =====================================================================
async def equipes_et_comptes(parc: list[str]) -> tuple[user, user, team, team]:
    """Deux équipes aux habilitations différentes, avec un modèle commun."""
    titre("2. Équipes, habilitations et héritage")

    data = team("DATA-01", "Data Platform", "data@entreprise.fr")
    secu = team("SEC-01", "Cybersécurité", "secu@entreprise.fr")
    await data.create_team()
    await secu.create_team()

    # Le premier modèle est accordé aux deux équipes : c'est lui qui
    # permet de vérifier que l'union ne produit pas de doublon.
    commun = parc[0]
    propre_data = parc[1] if len(parc) > 1 else None
    propre_secu = parc[2] if len(parc) > 2 else None

    await data.add_model(commun)
    await secu.add_model(commun)
    if propre_data:
        await data.add_model(propre_data)
    if propre_secu:
        await secu.add_model(propre_secu)

    refus = await data.add_model("inconnu:1b")
    print(f"  modèle hors catalogue : {refus.message}")

    # print(f"  DATA-01 : {[m['code_model'] for m in await data.load_models()]}")
    print(f"  SEC-01  : {[m['code_model'] for m in await secu.load_models()]}")

    # jdoe = user("jdoe", "MotDePasse!2026", "karim@entreprise.fr")
    # await karim.create_user("DATA-01")
    # await karim.join_team("SEC-01")

    # jdoe = user("jdoe", "Secret!2026", "jdoe@entreprise.fr")
    # await jdoe.create_user("DATA-01")

    # modeles_karim = [m["code_model"] for m in await karim.allowed_models()]
    modeles_jdoe = [m["code_model"] for m in await jdoe.allowed_models()]

    # print(f"\n  karim (DATA-01 + SEC-01) : {modeles_karim}")
    print(f"  jdoe         : {modeles_jdoe}")

    # Les attendus sont calculés, jamais écrits en dur : la démonstration
    # suit le parc installé sans qu'on ait à la retoucher.
    # attendu_karim = sorted({commun} | {m for m in (propre_data, propre_secu) if m})
    attendu_jdoe = sorted({commun} | ({propre_data} if propre_data else set()))
    #assert sorted(modeles_karim) == attendu_karim, modeles_karim
    assert sorted(modeles_jdoe) == attendu_jdoe, modeles_jdoe

    # habilitations = 2 + bool(propre_data) + bool(propre_secu)
    # print(f"  union sans doublon : {len(modeles_karim)} modèle(s) "
    #       f"pour {habilitations} habilitations")

    # if propre_secu:
    #     print(f"  {propre_secu} autorisé pour karim : "
    #           f"{await karim.can_use_model(propre_secu)}")
    #     print(f"  {propre_secu} autorisé pour jdoe  : "
    #           f"{await jdoe.can_use_model(propre_secu)}")

    return  jdoe,  secu


# =====================================================================
async def comptes_techniques(parc: list[str]) -> None:
    """Un agent tourne sous un compte de service, pas sous un compte humain."""
    titre("3. Comptes techniques")

    await team("AGENTS", "Comptes de service").create_team()
    await team("AGENTS", "").add_model(parc[0])

    compte, mot_de_passe = await user.creer_compte_technique(
        "agent-rag", "agent-rag@interne", "Agent RAG documentaire", "AGENTS"
    )
    print(f"  créé : {compte.email}")
    print(f"  mot de passe généré : {mot_de_passe[:12]}… (à archiver dans KeePass)")

    _, message = await user.creer_compte_technique("agent-x", "agent-x@interne", "   ")
    print(f"  sans description    : {message}")

    from Engine.models import authenticate

    print("\n  Un compte technique ne peut pas ouvrir de session :")
    # humain = await authenticate("karim@entreprise.fr", "MotDePasse!2026")
    technique = await authenticate("agent-rag@interne", mot_de_passe)
    # print(f"    humain    : {humain is not None}")
    print(f"    technique : {technique is not None}")

    print("\n  Inventaire :")
    for ligne in await user.comptes_techniques():
        print(f"    {ligne['login']:<12} {ligne['description']:<26} "
              f"équipes={ligne['equipes']} modèles={ligne['modeles']}")


# =====================================================================
async def concurrence() -> None:
    """Cinq créations lancées ensemble — l'intérêt du pool asynchrone."""
    titre("4. Écritures concurrentes")

    equipes = [team(f"TMP-{i:02d}", f"Équipe temporaire {i}") for i in range(5)]
    resultats = await asyncio.gather(*(t.create_team() for t in equipes))
    print(f"  {sum(1 for r in resultats if r)} / {len(resultats)} créations réussies")

    # Le même code, deux fois : ON CONFLICT garantit qu'aucun doublon ne
    # passe, même lancé en parallèle.
    doublons = await asyncio.gather(*(t.create_team() for t in equipes))
    print(f"  {sum(1 for r in doublons if not r)} / {len(doublons)} doublons refusés")


# =====================================================================
async def cycle_de_vie( jdoe: user, secu: team) -> None:
    """Ce qui survit à une suppression, et ce qui disparaît."""
    titre("5. Cycle de vie")

    # print(f"  membres de DATA-01 : {await data.load_members()}")

    # await secu.delete_member_team(karim)
    # print(f"  karim quitte SEC-01 → équipes : {await karim.load_teams()}")
    # print(f"  ses modèles deviennent : "
    #       f"{[m['code_model'] for m in await karim.allowed_models()]}")

    # print(f"\n  mot de passe vérifié : {await karim.check_password('MotDePasse!2026')}")

    # resultat = await data.delete_team()
    # print(f"\n  {resultat.message}")
    # print(f"  équipes de karim : {await karim.load_teams()}")
    print(f"  modèles de jdoe  : {[m['code_model'] for m in await jdoe.allowed_models()]}")
    print(f"  jdoe existe toujours : {await user.find('jdoe@entreprise.fr') is not None}")
    print(f"  le catalogue survit  : {len(await model_llm.catalogue())} modèle(s)")


# =====================================================================
async def main() -> None:
    await vider()
    try:
        parc, reel = await detecter_parc()
        await catalogue(parc, reel)
        jdoe,  secu = await equipes_et_comptes(parc)
        await comptes_techniques(parc)
        await concurrence()
        await cycle_de_vie( jdoe, secu)
        print("\n✓ Démonstration terminée.")
    finally:
        # Nettoyage même en cas d'échec : sans cela, une assertion ratée
        # laisse la base à moitié remplie, et la relance échoue sur des
        # « existe déjà » qui masquent la vraie cause.
        await vider()
        await close_pool()
        print("Base nettoyée, connexions fermées.")


if __name__ == "__main__":
    asyncio.run(main())
