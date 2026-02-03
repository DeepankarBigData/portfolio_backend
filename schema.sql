-- Enable pgvector
create extension if not exists vector;

-- Create chunks table (matches your screenshot)
create table if not exists public.chunks (
  id bigserial primary key,
  chunk_id uuid not null default gen_random_uuid(),
  chunk_hash text,
  document_title text,
  chunk_text text,
  meta jsonb,
  created_at timestamptz not null default now(),
  embedding vector(1024)
);
