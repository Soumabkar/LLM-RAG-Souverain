-- =====================================================================
-- Migration 004 : stockage vectoriel du RAG (pgvector)
--
-- Prérequis : une image PostgreSQL embarquant l'extension vector.
-- postgres:16-alpine ne l'a pas — passer à pgvector/pgvector:pg16.
--
-- La dimension (768) doit correspondre au modèle d'embedding :
-- nomic-embed-text produit des vecteurs de 768 composantes. Changer de
-- modèle impose de recréer la table et de réindexer tout le corpus.
--
--   docker compose exec -T postgres \
--       psql -U llm_admin -d llm_souverain_db < migrations/004_rag_pgvector.sql
-- =====================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS llm_souverain.rag_chunk (
    chunk_id   VARCHAR(128) PRIMARY KEY,
    doc_id     VARCHAR(64)  NOT NULL,
    idx        INTEGER      NOT NULL,
    texte      TEXT         NOT NULL,
    source     VARCHAR(512) NOT NULL,
    type_doc   VARCHAR(16)  NOT NULL,
    -- Contrôle d'accès : la recherche ne remonte que les chunks dont les
    -- équipes recoupent celles de l'appelant.
    equipes    TEXT[]       NOT NULL DEFAULT '{}',
    embedding  vector(768)  NOT NULL,
    indexe_le  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- HNSW : recherche approximative des plus proches voisins, en cosinus.
CREATE INDEX IF NOT EXISTS idx_rag_chunk_embedding
    ON llm_souverain.rag_chunk USING hnsw (embedding vector_cosine_ops);

-- GIN : accélère le filtre « equipes && ARRAY[...] ».
CREATE INDEX IF NOT EXISTS idx_rag_chunk_equipes
    ON llm_souverain.rag_chunk USING gin (equipes);

CREATE INDEX IF NOT EXISTS idx_rag_chunk_doc
    ON llm_souverain.rag_chunk (doc_id);

COMMENT ON TABLE llm_souverain.rag_chunk IS
    'Extraits vectorisés du RAG. Aucune donnée personnelle : masquées à l''étape data_processing.';

COMMIT;
