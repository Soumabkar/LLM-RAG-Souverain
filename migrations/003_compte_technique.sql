-- =====================================================================
-- Migration 003 : comptes techniques
--
-- Un agent s'exécute sous un compte dédié, distinct des comptes humains.
-- Sans ce marqueur, rien n'empêcherait quelqu'un de se connecter à
-- l'interface avec les identifiants d'un agent — et d'hériter de ses
-- habilitations.
--
--   docker compose exec -T postgres \
--       psql -U llm_admin -d llm_souverain_db < migrations/003_compte_technique.sql
-- =====================================================================

BEGIN;

SET search_path TO llm_souverain, public;

ALTER TABLE llm_souverain.user_llm
    ADD COLUMN IF NOT EXISTS technique   BOOLEAN      NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS description VARCHAR(256);

COMMENT ON COLUMN llm_souverain.user_llm.technique IS
    'true = compte de service porté par un agent ; interdit de connexion interactive';
COMMENT ON COLUMN llm_souverain.user_llm.description IS
    'À quoi sert ce compte — indispensable pour les comptes techniques';

-- Les comptes techniques sont peu nombreux face aux comptes humains :
-- un index partiel suffit et reste minuscule.
CREATE INDEX IF NOT EXISTS idx_user_llm_technique
    ON llm_souverain.user_llm (email) WHERE technique;

-- Vue d'inventaire : quel agent tourne sous quel compte, avec quels droits.
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

COMMIT;
