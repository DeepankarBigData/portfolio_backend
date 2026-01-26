-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create the chunks table
create table chunks (
  id bigserial primary key,
  document_title text,
  chunk_text text,
  chunk_tokens int,
  -- mxbai-embed-large has 1024 dimensions
  embedding vector(1024),
  meta jsonb
);

-- Optional: Create an HNSW index for faster similarity search
create index on chunks using hnsw (embedding vector_cosine_ops);
