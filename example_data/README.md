# SYNTHETIC DATA -- NOT REAL PATIENTS

Everything under `example_data/` is randomly generated (see
`utilities/generate_example_data.py`, fixed seed). No real patient, image or
clinical measurement went into producing it. It exists only to let someone
run `src/main.py` end to end without access to the real cohort, and to let
the "Expected data layout" section of the top-level README be checked
against a real run instead of taken on faith.

All IDs, TMA names and lot/batch labels use a "SYN" prefix on purpose, so
nothing here could be mistaken for real data if copied elsewhere. The numeric
values (intensities, clinical variables, survival times) have no biological
meaning -- they are random draws shaped to stay in plausible ranges so the
pipeline's numeric steps (GMM fitting, normalization, statistical tests)
don't choke on degenerate input, nothing more.

See the "Try it with synthetic data" section of the top-level
[README.md](../README.md) for exactly which `src/main.py` constants to point
here (including one -- `EXPECTED_BY_COHORT` -- that needs clearing rather
than repointing, since it holds the real cohort's reference thresholds).
