-- Initialize HE-300 Dashboard Database
-- This runs on first PostgreSQL container startup

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for common queries (Prisma handles table creation)
-- These will be applied after Prisma migrations run

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE he300_dashboard TO he300;
