# import psycopg2
# from psycopg2.extras import RealDictCursor
# # from ..database import get_db_connection

# def vector_to_literal(vec: list[float]) -> str:
#     """
#     Converts python list -> pgvector literal string.
#     Example: [0.1,0.2,...]
#     """
#     return "[" + ",".join(str(float(x)) for x in vec) + "]"

# def pg_vector_search(query_vec: list[float], k: int = 4) -> list[dict]:
#     """
#     Runs pgvector similarity search.
#     """
#     vec_lit = vector_to_literal(query_vec)

#     sql = """
#     SELECT chunk_id, document_title, chunk_text, meta,
#       1 - (embedding <#> %s::vector) AS score
#     FROM public.chunks
#     ORDER BY embedding <#> %s::vector
#     LIMIT %s;
#     """
#     # Note: I added chunk_tokens to select list since we might use it

#     try:
#         with get_db_connection() as conn:
#             with conn.cursor(cursor_factory=RealDictCursor) as cur:
#                 cur.execute(sql, (vec_lit, vec_lit, k))
#                 rows = cur.fetchall()
#         return rows

#     except psycopg2.Error as e:
#         raise RuntimeError(f"Database query failed: {str(e)}") from e
