-- =====================================================================
-- Migration 001 : passage de user_llm.code_team (1-N) à team_member (N-N)
--
-- À jouer UNIQUEMENT sur une base déjà initialisée avec l'ancien schéma.
-- Pour une base neuve, init/01_schema.sql suffit.
--
--   docker compose exec -T postgres \
--       psql -U llm_admin -d llm_souverain_db < migrations/001_team_member.sql
-- =====================================================================

BEGIN;

SET search_path TO llm_souverain, public;

CREATE TABLE IF NOT EXISTS llm_souverain.team_member (
    code_team   VARCHAR(256) NOT NULL,
    email       VARCHAR(256) NOT NULL,
    joined_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT pk_team_member PRIMARY KEY (code_team, email),
    CONSTRAINT fk_team_member_team FOREIGN KEY (code_team)
        REFERENCES llm_souverain.team_llm (code_team)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_team_member_user FOREIGN KEY (email)
        REFERENCES llm_souverain.user_llm (email)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_team_member_email
    ON llm_souverain.team_member (email);

-- Reprise de l'existant : chaque code_team non nul devient une adhésion.
INSERT INTO llm_souverain.team_member (code_team, email)
SELECT code_team, email
  FROM llm_souverain.user_llm
 WHERE code_team IS NOT NULL
ON CONFLICT DO NOTHING;

-- Contrôle : autant d'adhésions que d'utilisateurs rattachés avant migration.
DO $$
DECLARE
    nb_avant  INTEGER;
    nb_apres  INTEGER;
BEGIN
    SELECT count(*) INTO nb_avant
      FROM llm_souverain.user_llm WHERE code_team IS NOT NULL;
    SELECT count(*) INTO nb_apres FROM llm_souverain.team_member;
    IF nb_avant <> nb_apres THEN
        RAISE EXCEPTION 'Reprise incomplète : % lignes attendues, % obtenues', nb_avant, nb_apres;
    END IF;
    RAISE NOTICE 'Reprise OK : % adhésion(s) migrée(s)', nb_apres;
END
$$;

-- Suppression de l'ancienne colonne (source de vérité unique).
ALTER TABLE llm_souverain.user_llm DROP CONSTRAINT IF EXISTS fk_user_llm_team;
DROP INDEX IF EXISTS llm_souverain.idx_user_llm_code_team;
ALTER TABLE llm_souverain.user_llm DROP COLUMN IF EXISTS code_team;

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

COMMIT;
