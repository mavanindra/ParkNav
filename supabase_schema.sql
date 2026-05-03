create table if not exists public.parknav_state (
  key text primary key,
  value jsonb not null,
  updated_at timestamptz not null default now()
);

