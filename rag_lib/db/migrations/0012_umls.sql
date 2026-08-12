-- UMLS concept extraction columns for papers table.
-- umls_concepts_json:      extracted UMLS concepts (list of {cui, name, tui, ...})
-- umls_mapped_topics_json: OpenAlex topics mapped from UMLS concepts (list of {topic_id, display_name, similarity, ...})

ALTER TABLE papers ADD COLUMN umls_concepts_json TEXT;
ALTER TABLE papers ADD COLUMN umls_mapped_topics_json TEXT;
