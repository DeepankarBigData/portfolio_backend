# import hashlib




# # cache_key -> {"answer": ..., "sources": ..., "timings": ...}
# _answer_cache = {}
# _cache_max_size = 100

# def get_cache_key(query: str, k: int) -> str:
#     """
#     Creates stable cache key from query+k.
#     """
#     cache_string = f"{query.lower().strip()}:{k}"
#     return hashlib.md5(cache_string.encode()).hexdigest()

# def get_cached_answer(cache_key: str):
#     return _answer_cache.get(cache_key)

# def cache_answer(cache_key: str, answer: str, sources: list, timings: dict):
#     """
#     Adds result to cache.
#     if full, removes oldest entry.
#     """
#     if len(_answer_cache) >= _cache_max_size:
#         oldest_key = next(iter(_answer_cache))
#         del _answer_cache[oldest_key]

#     _answer_cache[cache_key] = {
#         "answer": answer,
#         "sources": sources,
#         "timings": timings,
#         "cached": True
#     }

# def get_cache_stats():
#     return {
#         "cache_size": len(_answer_cache),
#         "cache_max_size": _cache_max_size,
#         "cache_keys_preview": list(_answer_cache.keys())[:10]
#     }

# def clear_cache():
#     global _answer_cache
#     size = len(_answer_cache)
#     _answer_cache = {}
#     return size
