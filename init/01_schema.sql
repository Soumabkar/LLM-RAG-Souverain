-- =====================================================================
-- Schéma : llm_souverain
-- Joué automatiquement au premier démarrage du conteneur PostgreSQL
-- (répertoire /docker-entrypoint-initdb.d).
--
-- Modèle : un utilisateur peut appartenir à plusieurs équipes.
-- L'appartenance est portée par la table d'association team_member.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS llm_souverain;

SET search_path TO llm_souverain, public;

-- ---------------------------------------------------------------------
-- Table : team_llm
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_souverain.team_llm (
    code_team   VARCHAR(256) NOT NULL,
    team_name   VARCHAR(50)  NOT NULL,
    email       VARCHAR(256),
    CONSTRAINT pk_team_llm PRIMARY KEY (code_team)
);

COMMENT ON TABLE  llm_souverain.team_llm           IS 'Équipes rattachées à la plateforme LLM souverain';
COMMENT ON COLUMN llm_souverain.team_llm.code_team IS 'Identifiant fonctionnel de l''équipe';
COMMENT ON COLUMN llm_souverain.team_llm.email     IS 'Adresse de contact de l''équipe';

-- ---------------------------------------------------------------------
-- Table : user_llm
-- Plus de colonne code_team : l'appartenance vit dans team_member.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_souverain.user_llm (
    login       VARCHAR(50)  NOT NULL,
    password    VARCHAR(256) NOT NULL,
    email       VARCHAR(256) NOT NULL,
    -- Un compte technique porte un agent : il ne doit jamais pouvoir
    -- ouvrir une session interactive.
    technique   BOOLEAN      NOT NULL DEFAULT false,
    description VARCHAR(256),
    CONSTRAINT pk_user_llm       PRIMARY KEY (email),
    CONSTRAINT uq_user_llm_login UNIQUE (login)
);

CREATE INDEX IF NOT EXISTS idx_user_llm_technique
    ON llm_souverain.user_llm (email) WHERE technique;

COMMENT ON TABLE  llm_souverain.user_llm          IS 'Comptes utilisateurs';
COMMENT ON COLUMN llm_souverain.user_llm.email    IS 'Clé primaire : identifiant unique du compte';
COMMENT ON COLUMN llm_souverain.user_llm.password IS 'Hash du mot de passe (bcrypt/argon2), jamais en clair';

-- ---------------------------------------------------------------------
-- Table : team_member  (association N-N)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_souverain.team_member (
    code_team   VARCHAR(256) NOT NULL,
    email       VARCHAR(256) NOT NULL,
    joined_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT pk_team_member PRIMARY KEY (code_team, email),
    CONSTRAINT fk_team_member_team FOREIGN KEY (code_team)
        REFERENCES llm_souverain.team_llm (code_team)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT fk_team_member_user FOREIGN KEY (email)
        REFERENCES llm_souverain.user_llm (email)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

-- La PK couvre déjà (code_team, email) ; cet index sert aux requêtes
-- « à quelles équipes appartient cet utilisateur ? ».
CREATE INDEX IF NOT EXISTS idx_team_member_email
    ON llm_souverain.team_member (email);

COMMENT ON TABLE llm_souverain.team_member IS
    'Appartenance N-N entre user_llm et team_llm. Suppression en cascade des deux côtés.';

-- ---------------------------------------------------------------------
-- Vue de confort : utilisateurs et leurs équipes
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW llm_souverain.v_user_teams AS
SELECT u.email,
       u.login,
       coalesce(
           array_agg(tm.code_team ORDER BY tm.code_team)
               FILTER (WHERE tm.code_team IS NOT NULL),
           '{}'
       ) AS teams
  FROM llm_souverain.user_llm u
  LEFT JOIN llm_souverain.team_member tm ON tm.email = u.email
 GROUP BY u.email, u.login;

-- ---------------------------------------------------------------------
-- search_path par défaut pour le rôle applicatif
-- ---------------------------------------------------------------------
DO $$
BEGIN
    EXECUTE format(
        'ALTER ROLE %I IN DATABASE %I SET search_path TO llm_souverain, public',
        current_user, current_database()
    );
END
$$;

-- ---------------------------------------------------------------------
-- Table : model_llm  (catalogue des modèles)
--
-- code_model est l'identifiant technique passé au serveur d'inférence
-- (« llama3.1:8b »), display_name ce que voit l'utilisateur.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_souverain.model_llm (
    code_model    VARCHAR(128) NOT NULL,
    display_name  VARCHAR(50)  NOT NULL,
    description   VARCHAR(256),
    active        BOOLEAN      NOT NULL DEFAULT true,
    CONSTRAINT pk_model_llm PRIMARY KEY (code_model)
);

COMMENT ON COLUMN llm_souverain.model_llm.code_model IS
    'Identifiant technique transmis au serveur d''inférence';
COMMENT ON COLUMN llm_souverain.model_llm.active IS
    'false retire le modèle de toutes les équipes sans casser les habilitations';

-- ---------------------------------------------------------------------
-- Table : team_model  (association N-N équipe / modèle)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_souverain.team_model (
    code_team   VARCHAR(256) NOT NULL,
    code_model  VARCHAR(128) NOT NULL,
    granted_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT pk_team_model PRIMARY KEY (code_team, code_model),
    CONSTRAINT fk_team_model_team FOREIGN KEY (code_team)
        REFERENCES llm_souverain.team_llm (code_team)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_team_model_model FOREIGN KEY (code_model)
        REFERENCES llm_souverain.model_llm (code_model)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_team_model_model
    ON llm_souverain.team_model (code_model);

COMMENT ON TABLE llm_souverain.team_model IS
    'Modèles autorisés pour une équipe. Un utilisateur hérite de l''union des modèles de ses équipes.';

-- ---------------------------------------------------------------------
-- Vue : modèles accessibles par utilisateur
--
-- DISTINCT parce qu'un même modèle peut être accordé à plusieurs équipes
-- dont l'utilisateur fait partie : il ne doit apparaître qu'une fois.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW llm_souverain.v_user_models AS
SELECT DISTINCT
       tm.email,
       m.code_model,
       m.display_name,
       m.description
  FROM llm_souverain.team_member tm
  JOIN llm_souverain.team_model  t ON t.code_team = tm.code_team
  JOIN llm_souverain.model_llm   m ON m.code_model = t.code_model
 WHERE m.active;

-- ---------------------------------------------------------------------
-- Vue : inventaire des comptes techniques
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW llm_souverain.v_comptes_techniques AS
SELECT u.email,
       u.login,
       u.description,
       coalesce(
           array_agg(DISTINCT t.code_team) FILTER (WHERE t.code_team IS NOT NULL),
           '{}'
       ) AS equipes,
       coalesce(
           array_agg(DISTINCT m.code_model) FILTER (WHERE m.code_model IS NOT NULL),
           '{}'
       ) AS modeles
  FROM llm_souverain.user_llm u
  LEFT JOIN llm_souverain.team_member t ON t.email = u.email
  LEFT JOIN llm_souverain.team_model  x ON x.code_team = t.code_team
  LEFT JOIN llm_souverain.model_llm   m ON m.code_model = x.code_model AND m.active
 WHERE u.technique
 GROUP BY u.email, u.login, u.description;
