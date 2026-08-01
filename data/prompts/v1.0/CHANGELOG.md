# Prompt Bundle Changelog

## v1.0 — 2026-03-15

Initial versioned prompt bundle.

### Contents
- `system_prompt.txt` — ETL generation instructions: schema requirements, helper usage, county-clip pattern, condo collapse, exemption logic, upload step
- `playbook.md` — Snapshot of `docs/add-city-playbook.md`
- `schema.md` — Snapshot of `docs/parcel-parquet-format.md`
- `model.txt` — Pinned to `claude-sonnet-4-6`
- `few_shot/baltimore.py` — Reference: city-only ArcGIS source, USEGROUP+SDATCODE categorization, owner-name exempt detection, condo collapse on BLOCKLOT
- `few_shot/cincinnati.py` — Reference: county-wide ArcGIS source → osmnx city clip, CLASS-code categorization, GRPPCLID condo collapse

### Model pinning rationale
Pinned to `claude-sonnet-4-6` for cost/quality balance. ETL generation requires strong code output and structured reasoning but not Opus-level reasoning for most cities. Upgrade to Opus if generation quality regresses on complex cities.

### Change process
- Minor text fixes: edit in place, bump CHANGELOG, keep version directory
- Schema or playbook updates: new version directory, bump `CURRENT_VERSION`, run regression suite
- Model upgrade: new version directory, update `model.txt`, run full regression comparison
