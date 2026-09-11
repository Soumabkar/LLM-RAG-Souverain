-- =====================================================================
-- Migration 002 : modèles autorisés par équipe
--
-- À jouer sur une base en version 1.1 (schéma avec team_member).
-- Pour une base neuve, init/01_schema.sql contient déjà tout.
--
--   docker compose exec -T postgres \
--       psql -U llm_admin -d llm_souverain_db < migrations/002_team_model.sql
-- =====================================================================

BEGIN;

SET search_path TO llm_souverain, public;

CREATE TABLE IF NOT EXISTS llm_souverain.model_llm (
    code_model    VARCHAR(128) NOT NULL,
    display_name  VARCHAR(50)  NOT NULL,
    description   VARCHAR(256),
    active        BOOLEAN      NOT NULL DEFAULT true,
    CONSTRAINT pk_model_llm PRIMARY KEY (code_model)
);

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
-- Amorçage : un catalogue minimal et un accès pour toutes les équipes
-- existantes, pour ne pas couper le service pendant la migration.
-- Restreindre ensuite équipe par équipe.
-- ---------------------------------------------------------------------
INSERT INTO llm_souverain.model_llm (code_model, display_name, description)
VALUES ('llama3.1:8b', 'Llama 3.1 8B', 'Modèle généraliste, réponses rapides')
ON CONFLICT (code_model) DO NOTHING;

INSERT INTO llm_souverain.team_model (code_team, code_model)
SELECT code_team, 'llama3.1:8b' FROM llm_souverain.team_llm
ON CONFLICT DO NOTHING;

DO $$
DECLARE
    nb INTEGER;
BEGIN
    SELECT count(*) INTO nb FROM llm_souverain.team_model;
    RAISE NOTICE 'Habilitations créées : %', nb;
END
$$;

COMMIT;
