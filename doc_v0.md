marp: true
theme: gaia
paginate: true
size: 16:9
<!-- _class: lead -->
LLM-RAG Souverain
Socle LLM souverain, gouvernance par équipes et trajectoire Agents

Présentation Manager — Septembre 2026

1. Où en sommes-nous ?
Un socle LLM désormais structuré
PostgreSQL 16
Python asynchrone
Chainlit
Ollama / API OpenAI-compatible
Gestion utilisateurs
Gestion des équipes
Multi-appartenance utilisateur ↔ équipe
Sécurité des mots de passe
104 tests
Message clé

Le socle technique est construit.

2. Une évolution importante
Passage du modèle 1-N au modèle N-N
Avant
Utilisateur
    │
    └── 1 équipe

Maintenant
Utilisateur
    │
    ├── DATA
    ├── SECURITY
    └── R&D


Une table team_member porte désormais l'appartenance.

Bénéfice

La gouvernance peut maintenant être construite autour de l'équipe.

3. Architecture actuelle
Utilisateur
     │
     ▼
  Chainlit
     │
     ▼
Authentification
     │
     ▼
User / Team
     │
     ▼
  LLM Access
     │
     ▼
  AI Model
     │
     ▼
Ollama / OpenAI-compatible

Points forts
séparation des responsabilités ;
asynchronisme ;
streaming ;
health check ;
retry / timeout ;
historique par session.
4. Sécurité déjà intégrée
Défense en profondeur
bcrypt + pré-hachage SHA-256
hash factice contre l'énumération
SQL paramétré
validation des entrées
contraintes PostgreSQL
contrôle du compte avant accès LLM
sessions LLM isolées
104 tests

Sécurité + schéma + utilisateurs + équipes.

5. Ce qui manque encore
Aujourd'hui
Utilisateur
     ↓
Équipe
     ↓
Contexte
     ↓
Modèle global configuré

Cible
Utilisateur
     ↓
Équipe
     ↓
Policy
     ↓
Modèles autorisés
     ↓
Modèle sélectionné

C'est la prochaine étape de gouvernance.
6. Accès au modèle par équipe
Exemple cible
Équipe	Modèle local	Modèle avancé	Cloud
DATA	✓	✓	selon politique
R&D	✓	✓	✓
RH	✓	—	—
SECURITY	✓	✓	—
Règle serveur
Utilisateur ∈ Équipe
        +
Équipe → Modèle autorisé
        +
Modèle actif
        =
      ACCÈS

7. Base de données cible
USER
 │
 └── TEAM_MEMBER ── TEAM
                       │
                       ├── TEAM_MODEL_ACCESS
                       │             │
                       │             ▼
                       │           MODEL
                       │
                       └── CORPUS
                              │
                              └── DOCUMENT
                                      │
                                      └── CHUNK

Une même politique pour les modèles et les données.
8. RAG
Prochaine brique
Documents
    ↓
Ingestion
    ↓
Embeddings
    ↓
Vector Store
    ↓
Recherche
    ↓
ACL équipe
    ↓
Contexte
    ↓
LLM

Principe de sécurité

Un document interdit ne doit jamais entrer dans le contexte du LLM.

9. Vers l'Agent Loop
Première étape agentique
Question
   ↓
Plan
   ↓
Outil
   ↓
Observation
   ↓
Nouvelle action ?
   ├── Oui → boucle
   └── Non → réponse

Garde-fous
nombre maximum d'étapes ;
timeout ;
outils autorisés ;
budget tokens ;
arrêt sur erreur.
10. PydanticAI
Agents typés
Agent
 │
 ├── UserContext
 ├── TeamContext
 ├── RAG
 ├── Tools
 └── AgentResponse

Objectif

Faire du contexte de sécurité une donnée explicite du workflow et non une simple information du prompt.

11. LangGraph
Pour les workflows complexes
Authorize
    ↓
Retrieve
    ↓
Plan
    ↓
Tool
    ↓
Validate
    ├── Final
    ├── Continue
    └── Human

Apport
état ;
persistance ;
checkpoints ;
reprise ;
Human-in-the-Loop.
12. Trajectoire
        AUJOURD'HUI
             │
             ▼
      Socle LLM sécurisé
             │
             ▼
       Team → Model
             │
             ▼
        RAG sécurisé
             │
             ▼
        Agent Loop
             │
             ▼
         PydanticAI
             │
             ▼
          LangGraph

Une montée en puissance progressive.
13. Roadmap
Phase	Objectif
1	Sécurité / Keycloak
2	Team → Model
3	RAG + ACL
4	Agent Loop
5	PydanticAI
6	LangGraph + HITL
7	Observabilité + quotas
14. Valeur pour l'entreprise
Sécurité

Données et modèles gouvernés.

Souveraineté

Possibilité d'utiliser des modèles locaux.

Gouvernance

Les droits sont liés à l'organisation.

Industrialisation

Architecture évolutive vers RAG et agents.

Maîtrise

Quotas, audit et politiques peuvent être ajoutés progressivement.

15. Conclusion
Une IA gouvernée par les droits
IDENTITÉ
   ↓
ÉQUIPE
   ↓
DROITS
   ↓
MODÈLE
   ↓
DONNÉES
   ↓
OUTILS
   ↓
AGENT


L'IA ne doit pouvoir faire que ce que l'utilisateur et son organisation l'autorisent à faire.

Proposition

Valider la trajectoire :

Socle → Gouvernance → RAG → Agents → Industrialisation