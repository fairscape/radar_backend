-- 0031_abstract_source.sql — where a paper's abstract came from.
--
-- An uploaded PDF whose paper OpenAlex has no abstract for keeps the
-- abstract printed in the PDF (rag_lib.pdf_text.find_abstract), on the
-- uploader's own version only (FUNCTIONAL_SPEC P.3-3, P.5 Q1-Q2):
--   'pdf'  read from the uploaded file; the page says so
-- NULL for every other abstract (OpenAlex's, a Prosopia record's).
ALTER TABLE papers ADD COLUMN abstract_source TEXT;
