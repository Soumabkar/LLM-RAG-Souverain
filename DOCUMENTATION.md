# Plateforme LLM souverain — socle utilisateurs et équipes

Documentation fonctionnelle et technique du schéma `llm_souverain` et de la
couche d'accès Python.

| | |
|---|---|
| Version du schéma | 1.1 (association N-N) |
| Couche d'accès | asynchrone (asyncio) |
| Base | PostgreSQL 16 |
| Runtime | Python 3.11+ |
| Dernière validation | 104 tests, PostgreSQL 16.15 |

---

## 1. Documentation fonctionnelle

### 1.1 Objet

Le socle gère **qui a le droit d'utiliser la plateforme** et **au nom de
quelle équipe**. Il ne gère ni les conversations, ni les documents du RAG,
ni les quotas : ces briques viendront s'y rattacher.

Deux objets métier :

- **Utilisateur** — une personne physique disposant d'un compte. Identifiée
  par son adresse email.
- **Équipe** — une entité de rattachement (direction, projet, centre de
  coût). Identifiée par un code fonctionnel.

Un utilisateur peut appartenir à **plusieurs équipes** ; une équipe compte
**plusieurs membres**.

### 1.2 Règles de gestion

| Réf | Règle |
|---|---|
| RG-01 | L'email identifie l'utilisateur de façon unique. Deux comptes ne peuvent pas partager la même adresse. |
| RG-02 | Le login est unique lui aussi : il sert à l'authentification. |
| RG-03 | Le mot de passe n'est **jamais** stocké en clair. Seule son empreinte est conservée. |
| RG-04 | Un compte peut exister sans équipe. Il est alors créé, mais sans rattachement. |
| RG-05 | On ne peut rattacher un utilisateur qu'à une équipe existante. |
| RG-06 | Un utilisateur ne peut pas être deux fois membre de la même équipe. |
| RG-07 | Supprimer une équipe ne supprime pas ses membres : leurs comptes survivent, ils perdent seulement ce rattachement. |
| RG-08 | Supprimer un compte le retire automatiquement de toutes ses équipes. |
| RG-09 | Changer l'email d'un compte conserve ses rattachements. |
| RG-10 | Le code d'une équipe est immuable après création. Pour le changer, créer une nouvelle équipe et y déplacer les membres. |

### 1.3 Cas d'usage

#### Créer un utilisateur

1. L'administrateur saisit login, mot de passe, email, et éventuellement un
   code équipe.
2. Le système contrôle le format et la longueur des champs.
3. Si un code équipe est fourni et qu'il n'existe pas → refus, **rien n'est
   créé**.
4. Si l'email est déjà pris → refus, message « existe déjà ».
5. Si le login est déjà pris → refus, message dédié.
6. Sinon le compte est créé et rattaché, en une seule transaction.

#### Rattacher / détacher un membre

Le rattachement est réversible et sans effet sur le compte. Détacher un
membre de sa dernière équipe le laisse actif mais sans rattachement — c'est
volontaire : la désactivation d'un compte est une autre fonctionnalité.

#### Supprimer une équipe

L'opération indique le nombre d'adhésions retirées, pour que l'IHM puisse
demander confirmation avant une suppression à fort impact.

### 1.4 Messages retournés

Toutes les opérations renvoient un couple **succès / message**. Les messages
sont rédigés en français et destinés à être affichés tels quels.

| Situation | Message |
|---|---|
| Succès de création | `Utilisateur 'x@y.fr' créé.` |
| Email déjà pris | `L'utilisateur 'x@y.fr' existe déjà.` |
| Login déjà pris | `Le login 'jdoe' est déjà utilisé.` |
| Équipe inconnue | `Le code équipe 'DATA-99' n'existe pas.` |
| Champ vide | `Le champ login est obligatoire.` |
| Champ trop long | `Le champ login fait 60 caractères (maximum 50).` |
| Adhésion en double | `'x@y.fr' est déjà membre de l'équipe 'DATA-01'.` |
| Suppression d'équipe | `Équipe 'DATA-01' supprimée, 2 adhésion(s) retirée(s).` |

### 1.5 Hors périmètre

Authentification (déléguée à Keycloak), gestion des rôles et habilitations,
quotas de consommation, journalisation des usages, désactivation
(*soft delete*) des comptes.

---

## 2. Documentation technique

### 2.1 Arborescence

```
llm-souverain-db/
├── docker-compose.yml          PostgreSQL 16 + pgAdmin
├── .env.example                variables d'environnement
├── requirements.txt
├── pytest.ini
├── init/
│   └── 01_schema.sql           schéma complet (base neuve)
├── migrations/
│   └── 001_team_member.sql     1-N → N-N (base existante)
├── app/
│   ├── db.py                   pool de connexions
│   ├── security.py             hachage des mots de passe
│   ├── models.py               classes user et team
│   ├── ai_model.py             client LLM (AsyncOpenAI)
│   ├── chainlit_app.py         interface de chat
│   └── demo.py                 scénario de bout en bout
├── tests/                      104 tests
└── docs/
    └── DOCUMENTATION.md
```

### 2.2 Modèle de données

```
   team_llm                team_member                 user_llm
┌──────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ code_team PK │◄─────┤ code_team    PK  │─────►│ email        PK  │
│ team_name    │  1:N │ email        PK  │ N:1  │ login    UNIQUE  │
│ email        │      │ joined_at        │      │ password         │
└──────────────┘      └──────────────────┘      └──────────────────┘
                       ON DELETE CASCADE
                       ON UPDATE CASCADE
```

| Table | Colonne | Type | Contrainte |
|---|---|---|---|
| `team_llm` | `code_team` | `varchar(256)` | PK |
| | `team_name` | `varchar(50)` | NOT NULL |
| | `email` | `varchar(256)` | nullable |
| `user_llm` | `email` | `varchar(256)` | PK |
| | `login` | `varchar(50)` | UNIQUE, NOT NULL |
| | `password` | `varchar(256)` | NOT NULL, empreinte |
| `team_member` | `code_team` | `varchar(256)` | PK composite, FK → `team_llm` |
| | `email` | `varchar(256)` | PK composite, FK → `user_llm` |
| | `joined_at` | `timestamptz` | `DEFAULT now()` |

**Pourquoi une table d'association.** La colonne `user_llm.code_team` de la
version 1.0 a été supprimée. La conserver *en plus* de `team_member` aurait
créé deux sources de vérité pour la même information, condamnées à diverger.

**Pourquoi l'email en clé primaire.** C'est le choix fonctionnel retenu. Il
a un coût : un changement d'adresse est une modification de clé primaire,
propagée par `ON UPDATE CASCADE`. Tant que `team_member` est la seule table
fille, c'est tenable. Dès qu'apparaîtront les conversations, les logs et les
quotas — tous porteurs de l'email — une clé technique (`id BIGINT` ou `UUID`)
avec `email UNIQUE` deviendra préférable.

**Index.** La clé primaire de `team_member` couvre `(code_team, email)`, donc
la recherche « qui est dans cette équipe ». L'index `idx_team_member_email`
couvre le sens inverse, « à quelles équipes appartient cet utilisateur »,
que la PK ne sert pas.

**Vue `v_user_teams`.** Agrège les équipes par utilisateur sous forme de
tableau, pour éviter un `GROUP BY` dans chaque appelant.

### 2.3 Couche d'accès

#### `db.py`

Pool `psycopg_pool.AsyncConnectionPool` créé une fois par processus,
`dict_row` comme *row factory*, et `search_path` positionné sur
`llm_souverain, public` à l'ouverture de chaque connexion.

Le bloc `async with get_connection()` **valide la transaction en sortie
normale et fait un rollback si une exception remonte**. C'est ce qui rend
`create_user` atomique sans code de transaction explicite.

```python
async with get_connection() as conn, conn.cursor() as cur:
    await cur.execute("SELECT ...", (param,))
    ligne = await cur.fetchone()
```

Trois contraintes propres au pool asynchrone :

- **`open=False` est obligatoire.** Un `AsyncConnectionPool` ne peut pas
  s'ouvrir dans son constructeur : il n'y a pas encore de boucle
  d'événements à ce moment-là. D'où `await pool.open(wait=True)` dans
  `get_pool()`.
- **Le pool est lié à sa boucle.** Il ne survit pas à un changement de
  boucle d'événements — d'où `asyncio_default_test_loop_scope = session`
  dans `pytest.ini`, sans quoi chaque test repartirait sur une boucle neuve
  et un pool inutilisable.
- **Un `asyncio.Lock` protège la création.** Deux coroutines démarrant
  simultanément créeraient sinon chacune leur pool, et l'un des deux
  fuirait.
- **Sous Windows, la boucle par défaut ne convient pas.** Depuis Python
  3.8, Windows utilise `ProactorEventLoop` ; psycopg en mode asynchrone
  exige `SelectorEventLoop` et refuse sinon toute connexion. `db.py`
  bascule la politique dès l'import, ce qui couvre la démo, Chainlit et
  pytest sans intervention de l'appelant.

En cas d'échec à l'ouverture, le pool est fermé et `_pool` remis à `None` :
sans ce nettoyage, un pool mort resterait en cache et toutes les opérations
suivantes échoueraient sur `PoolClosed`, masquant l'erreur d'origine.

**Variables d'environnement nettoyées au `.strip()`.** Un `.env` enregistré
sous Windows (CRLF) laisse un `\r` en fin de valeur, qui produit un
`failed to resolve host 'localhost\r'` ou un échec d'authentification
parfaitement invisible à la lecture.

Configuration par variables d'environnement, `.env` chargé si
`python-dotenv` est présent :

| Variable | Défaut |
|---|---|
| `POSTGRES_HOST` | `localhost` |
| `POSTGRES_PORT` | `5432` |
| `POSTGRES_DB` | `llm_souverain_db` |
| `POSTGRES_USER` | `llm_admin` |
| `POSTGRES_PASSWORD` | `changeme` |
| `POSTGRES_POOL_MAX` | `10` |

#### `security.py`

Format stocké : `bcrypt-sha256$<hash bcrypt>`, soit 74 caractères.

Le mot de passe est pré-haché en SHA-256 puis encodé en base64 — 44 octets
constants — avant d'être passé à bcrypt. Ce détour contourne la **limite de
72 octets** de bcrypt, qui n'est pas théorique : 40 caractères accentués la
dépassent, puisqu'ils comptent double en UTF-8.

`verify_password` accepte les deux formats. `needs_rehash` signale un hash
historique, à remplacer lors de la prochaine connexion réussie :

```python
if verify_password(saisie, stocke) and needs_rehash(stocke):
    u.set_password(saisie)
    u.update_user()
```

#### `models.py`

Toutes les méthodes qui touchent la base sont des **coroutines** : elles
s'appellent avec `await`. Les helpers purement mémoire (`set_password`,
la propriété `code_team`) restent synchrones.

Elles renvoient un `Result(ok, message, data)`, utilisable directement dans
un `if` grâce à `__bool__`. Les exceptions `psycopg` sont capturées et
converties ; aucune erreur base ne remonte à l'appelant.

```python
equipe = team("DATA-01", "Data Platform")
if await equipe.create_team():
    await equipe.add_member_team(karim)

# Opérations indépendantes : les lancer ensemble plutôt qu'en série
resultats = await asyncio.gather(*(t.create_team() for t in equipes))
```

| Classe | Méthode | Effet |
|---|---|---|
| `user` | `create_user(code_team=None)` | crée le compte, rattache si code fourni |
| | `delete_user()` | supprime le compte et ses adhésions |
| | `update_user()` | met à jour login et mot de passe |
| | `change_email(new)` | change la clé primaire |
| | `join_team(code)` / `leave_team(code)` | rattachement, détachement |
| | `load_teams()` / `is_member_of(code)` | lecture des adhésions |
| | `check_password(clair)` | vérifie contre l'empreinte stockée |
| | `user.load(email)` / `user.find(email)` | chargement |
| `team` | `create_team()` | insère l'équipe |
| | `update_team()` | met à jour nom et email |
| | `delete_team()` | supprime l'équipe, cascade sur les adhésions |
| | `add_member_team(u)` / `delete_member_team(u)` | gestion des membres |
| | `load_members()` | liste des emails |
| | `team.load(code)` | chargement complet |

Deux points de conception à connaître avant de modifier ce fichier :

- **Contrôle d'existence atomique.** Les créations utilisent
  `INSERT ... ON CONFLICT DO NOTHING` et testent `rowcount`, plutôt qu'un
  `SELECT` suivi d'un `INSERT`. Comportement identique côté appelant, mais
  deux appels concurrents ne peuvent pas créer le même email.
- **`update_team()` et l'email.** `self.email is None` laisse la colonne
  inchangée (`COALESCE`). Pour vider l'adresse, passer une chaîne vide.

### 2.4 Installation

```bash
cp .env.example .env          # puis modifier les mots de passe
docker compose up -d
pip install -r requirements.txt
python -m app.demo
```

Les scripts de `init/` ne s'exécutent qu'au **premier** démarrage, quand le
volume `pgdata` est vide. Après modification du schéma :

```bash
docker compose down -v && docker compose up -d
```

Sur une base déjà remplie en version 1.0, utiliser la migration :

```bash
docker compose exec -T postgres \
    psql -U llm_admin -d llm_souverain_db < migrations/001_team_member.sql
```

Elle reprend les `code_team` existants, **vérifie que le compte tombe juste**
et fait échouer la transaction sinon, puis supprime l'ancienne colonne.

### 2.5 Tests

```bash
pytest                 # tout
pytest -m "not db"     # unitaires seuls, sans PostgreSQL
pytest --cov=app       # avec couverture
```

104 tests répartis en quatre modules :

| Module | Portée | Couvre |
|---|---|---|
| `test_security.py` | unitaire, sans base | hachage, limite 72 octets, compatibilité ascendante |
| `test_schema.py` | intégration | contraintes, cascades, `search_path`, vue |
| `test_user.py` | intégration | cycle de vie du compte, validation, concurrence |
| `test_team.py` | intégration | cycle de vie de l'équipe, multi-appartenance |

Les tests d'intégration sont marqués `db` et sont **ignorés** si PostgreSQL
n'est pas joignable, plutôt que mis en échec. Chaque test part d'une base
vide (`TRUNCATE` avant et après).

### 2.6 Sécurité

Acquis :

- mot de passe haché en bcrypt (coût 12), jamais journalisé ;
- requêtes systématiquement paramétrées — aucune concaténation SQL ;
- validation des longueurs en amont de la base ;
- messages d'erreur qui ne divulguent pas d'information sur les hash.

À traiter avant mise en production :

- `POSTGRES_PASSWORD` en clair dans `.env` → passer par un gestionnaire de
  secrets ;
- pgAdmin exposé sans TLS et en `SERVER_MODE: False` → à réserver au poste
  de développement ;
- port 5432 publié sur l'hôte → à restreindre au réseau Docker en production ;
- pas de politique de complexité des mots de passe ni de limitation des
  tentatives — à traiter au niveau de Keycloak.

### 2.7 Limites connues

| Limite | Impact | Piste |
|---|---|---|
| Email en clé primaire | changement d'adresse = mise à jour en cascade | clé technique + `email UNIQUE` |
| Pas de rôle dans `team_member` | impossible de distinguer un responsable d'équipe | ajouter une colonne `role` |
| Pas de *soft delete* | suppression définitive, sans historique | colonne `deleted_at` |
| `load_members()` charge tout | coûteux au-delà de quelques milliers de membres | pagination |
| Un aller-retour par méthode | `add_member_team` recharge ensuite les deux collections | rendre le rechargement optionnel |

### 2.8 Port d'écoute

Le conteneur publie **5433** sur l'hôte, pas 5432. Une installation
PostgreSQL native occupe fréquemment le port standard sur un poste de
développement, et elle intercepte alors silencieusement les connexions
destinées au conteneur — le symptôme est un échec d'authentification
inexplicable, malgré un mot de passe correct.

Indice permettant de trancher : le conteneur Alpine tourne en `en_US.utf8`
et ne peut produire que des messages d'erreur en anglais. Un
`FATAL : authentification par mot de passe échouée` en français vient donc
forcément d'un autre serveur.

Depuis pgAdmin en revanche, l'hôte est `postgres` et le port `5432` : ce
sont les valeurs internes au réseau Docker, indépendantes de la
publication vers l'hôte.
