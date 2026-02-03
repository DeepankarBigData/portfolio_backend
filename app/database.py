# import os
# import atexit
# import psycopg2
# from psycopg2.pool import SimpleConnectionPool
# from contextlib import contextmanager
# from urllib.parse import urlparse, urlunparse, quote_plus
# from .config import DATABASE_URL

# # ============================================================
# # DATABASE URL VALIDATION + FIXING
# # ============================================================

# def validate_and_fix_database_url(url: str) -> str | None:
#     """
#     Validates DATABASE_URL format and reconstructs it safely.
#     """
#     if not url:
#         return None

#     # Must start with correct PostgreSQL protocol
#     if not url.startswith(("postgresql://", "postgres://")):
#         return None

#     try:
#         parsed = urlparse(url)

#         # Must have hostname
#         if not parsed.hostname:
#             return None

#         # If password is missing but URL has "@", it is likely malformed
#         if not parsed.password and "@" in url:
#             return None

#         # Build netloc properly
#         if parsed.password:
#             # URL encode password (handles special characters)
#             password_encoded = quote_plus(parsed.password)
#             netloc = f"{parsed.username}:{password_encoded}@{parsed.hostname}"
#         else:
#             netloc = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname

#         if parsed.port:
#             netloc += f":{parsed.port}"

#         # Ensure database path exists
#         fixed_url = urlunparse((
#             parsed.scheme,
#             netloc,
#             parsed.path or "/postgres",
#             parsed.params,
#             parsed.query,
#             parsed.fragment
#         ))

#         return fixed_url

#     except Exception as e:
#         print(f"Warning: Could not parse DATABASE_URL: {e}")
#         return None

# def build_database_url_if_missing() -> str:
#     """
#     Validates DATABASE_URL from environment.
#     """
#     # If user provided DATABASE_URL -> validate/fix it
#     if DATABASE_URL:
#         fixed_url = validate_and_fix_database_url(DATABASE_URL)
#         if fixed_url:
#             return fixed_url
#         else:
#             print(f"Warning: DATABASE_URL format may be incorrect: {DATABASE_URL[:50]}...")
#             print("Expected format: postgresql://user:password@host:port/database")

#     # Nothing exists -> fail fast
#     raise SystemExit(
#         "DATABASE_URL required in .env file.\n\n"
#         "Format: postgresql://postgres:[PASSWORD]@db.host:5432/postgres"
#     )

# # Ensure DATABASE_URL exists now
# FINAL_DATABASE_URL = build_database_url_if_missing()

# def ensure_db_url_has_required_params(url: str) -> str:
#     """
#     Ensures:
#     - sslmode=require (Supabase requires SSL)
#     - connect_timeout=10 (avoid long hangs)
#     """
#     if "sslmode" not in url:
#         sep = "&" if "?" in url else "?"
#         url = f"{url}{sep}sslmode=require"

#     if "connect_timeout" not in url:
#         sep = "&" if "?" in url else "?"
#         url = f"{url}{sep}connect_timeout=10"

#     return url

# def mask_database_url(url: str) -> str:
#     """
#     Prints safe URL without exposing password.
#     """
#     safe_url = url
#     if "@" in safe_url:
#         parts = safe_url.split("@")
#         if ":" in parts[0]:
#             user_pass = parts[0].split(":", 1)
#             safe_url = f"{user_pass[0]}:***@{parts[1]}"
#     return safe_url

# FINAL_DATABASE_URL = ensure_db_url_has_required_params(FINAL_DATABASE_URL)

# # Create connection pool (shared across requests)
# connection_pool = None
# try:
#     parsed = urlparse(FINAL_DATABASE_URL)
#     if not parsed.hostname:
#         raise ValueError("DATABASE_URL missing hostname")
#     if not parsed.username:
#         raise ValueError("DATABASE_URL missing username")

#     print(f"Connecting to database: {mask_database_url(FINAL_DATABASE_URL)}")
#     print(f"Connecting to hostname: {parsed.hostname}")

#     # Direct test connection (gives better startup error)
#     try:
#         print("Testing direct database connection...")
#         test_conn = psycopg2.connect(FINAL_DATABASE_URL, connect_timeout=15)
#         with test_conn.cursor() as cur:
#             cur.execute("SELECT 1")
#         test_conn.close()
#         print("Direct connection test successful")
#     except psycopg2.OperationalError as e:
#         raise SystemExit(f"Database connection failed: {str(e)}") from e

#     # Pool for production: prevents reconnect per request
#     connection_pool = SimpleConnectionPool(
#         minconn=1,
#         maxconn=10,
#         dsn=FINAL_DATABASE_URL
#     )

#     print("Database connection pool created successfully")

# except Exception as e:
#     raise SystemExit(f"Failed to create database connection pool: {str(e)}") from e


# @contextmanager
# def get_db_connection():
#     """
#     Context manager to safely:
#     - borrow connection from pool
#     - return it after use
#     """
#     conn = None
#     try:
#         conn = connection_pool.getconn()
#         yield conn
#     finally:
#         if conn:
#             connection_pool.putconn(conn)

# # When server stops, close all DB connections
# atexit.register(lambda: connection_pool.closeall() if connection_pool else None)
