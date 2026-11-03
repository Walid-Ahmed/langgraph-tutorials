-- Appendix F, Listings F.5 and F.6: a login role and a database it owns.
--
-- Run as a superuser:
--   macOS (Postgres.app / Homebrew):  psql -h localhost -d postgres -f 13-Postgres-Setup/01-create-database.sql
--   Linux:                            sudo -u postgres psql -f 13-Postgres-Setup/01-create-database.sql
--   Windows:                          psql -U postgres -h localhost -f 13-Postgres-Setup/01-create-database.sql
--
-- Expected output:
--   CREATE ROLE
--   CREATE DATABASE
--
-- Change the password before you use this anywhere but your own laptop, and
-- put the same password in DB_URI (.env.example).

-- A role that may log in, with a password. LOGIN is what separates a
-- user from a group; without it the role exists but can never connect.
CREATE ROLE langgraph WITH LOGIN PASSWORD 'change-me';

-- A database for the agent, owned by that role.
CREATE DATABASE langgraph_stm OWNER langgraph;

-- Listing F.6 — only if the role does NOT own the database (for example, a
-- database someone else created). Since PostgreSQL 15 the public schema is
-- not writable by everyone, so grant it explicitly, from inside that database.
-- Uncomment both lines to use it:
-- \c langgraph_stm
-- GRANT ALL ON SCHEMA public TO langgraph;
