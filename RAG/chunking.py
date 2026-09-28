"""Découpage du corpus en extraits.

Découpage récursif : on tente d'abord de couper entre paragraphes, puis
entre lignes, puis entre phrases, et en dernier recours entre mots. Un
chunk qui s'arrête au milieu d'une phrase produit un embedding flou et
une citation incompréhensible.

Le chevauchement répète la fin d'un chunk au début du suivant : une idée
qui tombe à cheval sur une frontière reste trouvable dans les deux.
"""

from __future__ import annotations

from .config import Chunk, TypeDocument

SEPARATEURS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def _decouper(texte: str, taille: int, separateurs: tuple[str, ...]) -> list[str]:
    if len(texte) <= taille:
        return [texte]

    separateur = next((s for s in separateurs if s in texte), "")
    if not separateur:
        # Aucun séparateur : coupe franche, dernier recours.
        return [texte[i:i + taille] for i in range(0, len(texte), taille)]

    suivants = separateurs[separateurs.index(separateur) + 1:]
    morceaux, courant = [], ""
    for partie in texte.split(separateur):
        candidat = f"{courant}{separateur}{partie}" if courant else partie
        if len(candidat) <= taille:
            courant = candidat
            continue
        if courant:
            morceaux.append(courant)
        # Une partie trop longue à elle seule est redécoupée plus finement.
        if len(partie) > taille:
            morceaux.extend(_decouper(partie, taille, suivants))
            courant = ""
        else:
            courant = partie
    if courant:
        morceaux.append(courant)
    return morceaux


def _chevaucher(morceaux: list[str], chevauchement: int) -> list[str]:
    """Préfixe chaque morceau par la fin du précédent, coupée sur un mot."""
    if chevauchement <= 0 or len(morceaux) < 2:
        return morceaux
    resultat = [morceaux[0]]
    for precedent, courant in zip(morceaux, morceaux[1:]):
        fin = precedent[-chevauchement:]
        espace = fin.find(" ")
        if 0 <= espace < len(fin) - 1:
            fin = fin[espace + 1:]  # ne pas commencer au milieu d'un mot
        resultat.append(f"{fin} {courant}".strip())
    return resultat


def decouper(
    texte: str,
    doc_id: str,
    source: str,
    type_doc: TypeDocument,
    equipes: list[str],
    taille: int = 1000,
    chevauchement: int = 150,
) -> list[Chunk]:
    """Découpe un texte en chunks porteurs de leurs métadonnées.

    Les équipes sont recopiées sur chaque chunk : c'est à ce niveau que
    la recherche filtrera, pas au niveau du document.
    """
    if chevauchement >= taille:
        raise ValueError("Le chevauchement doit être inférieur à la taille des chunks.")

    morceaux = [m.strip() for m in _decouper(texte, taille, SEPARATEURS) if m.strip()]
    morceaux = _chevaucher(morceaux, chevauchement)

    return [
        Chunk(
            chunk_id=f"{doc_id}-{i:05d}",
            doc_id=doc_id,
            index=i,
            texte=morceau,
            equipes=list(equipes),
            source=source,
            type=type_doc,
        )
        for i, morceau in enumerate(morceaux)
    ]
