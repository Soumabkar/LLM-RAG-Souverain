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
    CONSTRAINT pk_user_llm       PRIMARY KEY (email),
    CONSTRAINT uq_user_llm_login UNIQUE (login)
);

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
