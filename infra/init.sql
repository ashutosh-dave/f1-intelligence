-- F1 Race Intelligence Engine — Database Initialization
-- This runs automatically when the Postgres container starts for the first time.

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE f1_intelligence TO f1admin;
