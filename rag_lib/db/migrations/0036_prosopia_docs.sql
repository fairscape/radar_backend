-- A copy of each Prosopia profile an interest's papers were imported from,
-- researcher profile saved or not: its interests join the topic list
-- on the topic step (FUNCTIONAL_SPEC M.3, rag_lib/prosopia_interests.py).
ALTER TABLE profiles ADD COLUMN prosopia_docs_json TEXT;
