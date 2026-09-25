# No Alembic revisions here — on purpose

The tables CIRISBench reads and writes (`evaluations`, `frontier_models`, `agent_profiles`,
`tenant_tiers`) are **owned by CIRISNode**: `cirisnode/db/migrations/*.sql`, applied by
`cirisnode/db/migrator.py` at every CIRISNode start on every origin.

Until 2026-09 this directory held a parallel Alembic chain (001–004) that created and altered the
same tables. Two migration owners for one schema produced two different production shapes (US never
got `002_add_checkpoint_columns`; EU did, by hand), which broke every bench write and bench's own
crash-recovery pass on US for seven months. See CIRISAI/CIRISBench#8 and CIRISAI/CIRISNode#37.

Rules:

- **Do not add revisions for shared tables here.** A change to `engine/db/models.py` that needs a new
  column lands as a CIRISNode migration first (e.g. `022_bench_shared_columns.sql`), then the model.
- Bench-private tables (none today) may use this chain; keep them out of CIRISNode's table set.
- `alembic upgrade head` is no longer run by the container entrypoint or by CIRISCore's playbooks.
