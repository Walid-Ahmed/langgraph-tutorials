-- Appendix F, Listing F.7: a table of your own, and the four statements that
-- do almost everything (INSERT, SELECT, UPDATE, DELETE).
--
-- Run as the new role, after 01-create-database.sql:
--   psql -h localhost -p 5432 -U langgraph -d langgraph_stm -f 13-Postgres-Setup/02-crud-demo.sql
--
-- Expected output:
--   CREATE TABLE
--   INSERT 0 1
--   INSERT 0 2
--    id | thread_id |              body
--   ----+-----------+--------------------------------
--     1 | demo-1    | The user prefers metric units.
--     2 | demo-1    | The user is based in Toronto.
--   (2 rows)
--
--   UPDATE 1
--   DELETE 1
--    count
--   -------
--        2
--   (1 row)
--
-- The table stays so you can look at it in pgAdmin (Section F.4). Drop it when
-- you are done experimenting:  DROP TABLE notes;

CREATE TABLE notes (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    thread_id   TEXT        NOT NULL,
    body        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO notes (thread_id, body)
VALUES ('demo-1', 'The user prefers metric units.');

INSERT INTO notes (thread_id, body) VALUES
    ('demo-1', 'The user is based in Toronto.'),
    ('demo-2', 'Unrelated conversation.');

SELECT id, thread_id, body
FROM notes
WHERE thread_id = 'demo-1'
ORDER BY id;

UPDATE notes
SET body = 'The user prefers metric units and 24-hour time.'
WHERE id = 1;

DELETE FROM notes WHERE thread_id = 'demo-2';

SELECT count(*) FROM notes;
