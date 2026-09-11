"""Démonstration du modèle N-N en asynchrone : python demo_db_user_team.py"""

from __future__ import annotations

import asyncio
import logging

from Engine.db import close_pool
from Engine.models import model_llm, team, user

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


async def main() -> None:
    data = team("DATA-01", "Data Platform", "data@entreprise.fr")
    secu = team("SEC-01", "Cybersécurité", "secu@entreprise.fr")
    print(await data.create_team())
    print(await secu.create_team())

    karim = user("ksoumahoro", "MotDePasse!2026", "karim@entreprise.fr")
    print(await karim.create_user("DATA-01"))

    # Multi-appartenance
    print(await karim.join_team("SEC-01"))
    print(await karim.join_team("SEC-01"))         # déjà membre
    print(await karim.join_team("TEAM-INCONNUE"))  # équipe inexistante
    print("Équipes de karim :", await karim.load_teams())

    # Utilisateur sans équipe, rattaché après coup
    jdoe = user("jdoe", "Secret!2026", "jdoe@entreprise.fr")
    print(await jdoe.create_user())
    print(await data.add_member_team(jdoe))
    print("Membres DATA-01 :", await data.load_members())

    # -----------------------------------------------------------------
    # Modèles : catalogue, habilitations par équipe, héritage utilisateur
    # -----------------------------------------------------------------
    print("\n--- Catalogue ---")
    # verify=False : la démo doit tourner sans serveur d'inférence. En
    # exploitation, laisser le défaut — create_model() refuse alors un
    # modèle qui n'est pas installé sur le serveur.
    print(await model_llm("llama3.1:8b", "Llama 3.1 8B", "Généraliste rapide").create_model(verify=True))
    print(await model_llm("mistral:7b", "Mistral 7B", "Bon en français").create_model(verify=True))
    print(await model_llm("qwen2.5-coder:7b", "Qwen Coder", "Spécialisé code").create_model(verify=True))
    print(await model_llm("llama3.2:latest", "Llama 3.2", "Généraliste rapide").create_model(verify=True))
    print(await model_llm("qwen2.5:latest", "Qwen 2.5", "Spécialisé code").create_model(verify=True))
    

    # Ce qu'il se passe avec le contrôle actif et un modèle absent.
    async def faux_serveur(code: str) -> bool:
        return code in {"llama3.1:8b", "mistral:7b", "qwen2.5-coder:7b"}

    print(await model_llm("gpt-oss:120b", "GPT-OSS").create_model(verifier=faux_serveur))

    print("\n--- Habilitations ---")
    print(await data.add_model("llama3.2:latest"))
    print(await data.add_model("qwen2.5:latest"))   
    print(await secu.add_model("mistral:7b"))
    print(await secu.add_model("qwen2.5:latest"))      
    print(await data.add_model("inconnu:1b"))       # refusé : hors catalogue

    print("DATA-01 :", [m["code_model"] for m in await data.load_models()])
    print("SEC-01  :", [m["code_model"] for m in await secu.load_models()])

    print("\n--- Héritage par les équipes ---")
    modeles_karim = [m["code_model"] for m in await karim.allowed_models()]
    modeles_jdoe = [m["code_model"] for m in await jdoe.allowed_models()]
    print("karim (DATA-01 + SEC-01) :", modeles_karim)
    print("jdoe  (DATA-01)          :", modeles_jdoe)

    # llama3.1:8b est accordé aux deux équipes de karim : l'union le compte
    # une seule fois, c'est le DISTINCT de la vue v_user_models.
    # assert modeles_karim == ["llama3.1:8b", "mistral:7b", "qwen2.5-coder:7b"]
    # assert modeles_jdoe == ["llama3.1:8b", "qwen2.5-coder:7b"]
    # print("union sans doublon :", len(modeles_karim), "modèles pour 4 habilitations")

    print("mistral autorisé pour karim :", await karim.can_use_model("mistral:7b"))
    print("mistral autorisé pour jdoe  :", await jdoe.can_use_model("mistral:7b"))

    # Deux créations concurrentes : c'est tout l'intérêt de l'asynchrone.
    equipes = [team(f"TMP-{i:02d}", f"Équipe {i}") for i in range(5)]
    resultats = await asyncio.gather(*(t.create_team() for t in equipes))
    print("Créations concurrentes :", sum(1 for r in resultats if r), "/", len(resultats))

    # Retrait d'une seule adhésion
    print(await secu.delete_member_team(karim))
    print("Équipes de karim :", await karim.load_teams())

    # Vérification du mot de passe
    print("Mot de passe correct :", await karim.check_password("MotDePasse!2026"))

    # # Suppression d'équipe : les comptes survivent, les modèles se perdent
    # print("\n--- Suppression de DATA-01 ---")
    # print(await data.delete_team())
    # print("Équipes de karim après suppression :", await karim.load_teams())
    # print("Modèles de karim :", [m["code_model"] for m in await karim.allowed_models()])
    # print("Modèles de jdoe  :", [m["code_model"] for m in await jdoe.allowed_models()])
    # print("Le catalogue survit :", len(await model_llm.catalogue()), "modèles")
    # print("jdoe existe toujours :", await user.find("jdoe@entreprise.fr") is not None)

    # # Nettoyage
    # print(await karim.delete_user())
    # print(await jdoe.delete_user())
    # print(await secu.delete_team())
    # await asyncio.gather(*(t.delete_team() for t in equipes))
    # await asyncio.gather(
    #     *(
    #         model_llm(code, "").delete_model()
    #         for code in ("llama3.1:8b", "mistral:7b", "qwen2.5-coder:7b")
    #     )
    # )

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())