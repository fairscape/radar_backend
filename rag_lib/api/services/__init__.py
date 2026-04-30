"""Service layer.

Pure-function business logic for the API. Routers stay thin (parse
request → call service → return Pydantic model); services own the
DB-row → schema translation via ``rag_lib.api.mappers``.
"""
