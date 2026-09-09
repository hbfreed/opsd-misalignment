# Reproducing the study

## Regenerate the published tables offline

From this repository's root:

```bash
python3 scripts/reproduce.py
python3 scripts/reproduce.py --check
```

These commands need only the Python standard library (Python 3.10+). They do not load credentials, download a tokenizer, or call an API. The first regenerates the three marked table blocks in `README.md`, `results/tables.md`, and `results/metrics.csv`. It preserves the README's prose. `--check` validates without writing.

Validation checks the copied-file SHA-256 inventory, manifests and reference ordering, all 20 distillation completion records and saved configurations, training steps, checkpoint identities, all 26,400 response rows, and the 8,800 committed-answer grading receipts. It cross-checks the merged grades against the original grades plus the rejudging receipts, and verifies hedging independently. The tables include two untrained baselines, all 16 v2 cells, and only the four completed v3 follow-ups.

`results/metrics.csv` reports sampled, scored, incorrect, missing-correctness, hedged, scored-hedging, and missing-hedging counts separately. There are five completions on each of 40 trivia questions per evaluation condition. These are repeated completions, not 200 independent questions.

## Repository map

| Path | Purpose |
|---|---|
| `README.md` | Author's writeup, with tables regenerated from saved evidence |
| `results/` | Current tables and denominator-rich metrics |
| `scripts/reproduce.py` | Portable offline validation and table generation |
| `experiments/obvious_lies_cot_inkling/` | Patient creation, reference preparation, matched-grid training and evaluation |
| `experiments/obvious_lies_cot_inkling/study/` | Frozen v2/v3 manifests, reference pairs, run evidence, and final rejudging |
| `experiments/_shared/opsd.py` | Frozen-teacher top-20 forward-KL trainer |
| `inkling_renderer.py` | Inkling prompt/response rendering |
| `judges/prompts.py` | Broad-evaluation judge prompts |
| `provenance/source.json` | Exact source snapshot inventory and recorded release edits |
| `tests/` | Offline checks of experimental controls and evaluation handling |

The experiment's original directory name is retained to preserve imports and provenance. It is not a claim that a separate with-CoT patient experiment completed. Fish experiments, earlier exploratory arms, operational shell drivers, stale result narratives, and adapter binaries are outside this release.

## Running new experiments

Execution uses Python 3.11+ and the included dependency lock:

```bash
uv sync --locked
cp .env.example .env
```

Set `TINKER_API_KEY` for training/sampling and `OPENROUTER_API_KEY` for reference screening/judging. New execution uses paid services. Saved historical prices and model IDs are recorded experimental settings, not current quotes or availability guarantees. Offline table reproduction does not need these services.

The existing `tinker://` checkpoint pointers identify the actual patient and trained cells. They do not bundle checkpoint weights or establish access from another account. Adapter binaries remain in the original research workspace; no public adapter download is configured in this local release.

For a new grid, copy `study/settings.v1.json` (v2 design) or `study/settings.v3.json` (student-trigger follow-ups) into `new_runs/`, set checkpoint pointers accessible to your account, and generate a fresh manifest. The historical manifests contain original absolute paths and code hashes; they are evidence, not portable launch files. The offline reproduction command validates them without opening those original paths.

Example planning commands from the repository root, after creating `new_runs/settings.json`:

```bash
python3 experiments/obvious_lies_cot_inkling/grid_study.py plan \
  --pairs experiments/obvious_lies_cot_inkling/study/reference_verified.jsonl \
  --settings new_runs/settings.json \
  --out new_runs/manifest.json

python3 experiments/obvious_lies_cot_inkling/grid_study.py run \
  new_runs/manifest.json \
  --cell s-poisoned_t-poisoned_prompt-none_ref-good \
  --runs new_runs/runs
```

The second command previews the v2 cell. Adding `--execute` trains it. `grid_evaluate.py sample` and `grid_evaluate.py judge` expose evaluation subcommands; use `--help` for their arguments. `grid_report.py` generates reports for new manifests. These reports use their supplied grades; to compare to the README, use the committed-answer rubric in `study/rejudge_trivia.py`, which also measures hedging. The original broad/correctness rubric alone does not reproduce the README's trivia metric.

The v3 settings define 16 candidate cells, but this study ran only four. To reproduce its scope, select the four cell IDs listed as `version=v3` in `results/metrics.csv`. Running the entire v3 manifest with `run_grid.py --execute` would expand the study beyond those follow-ups.

`finetune_tinker.py` contains patient training; `make_opsd_split.py`, `make_selfref.py`, and `verify_references.py` record input preparation. Inspect their command-line options and output paths before use: some preparation scripts write their fixed output filenames and `make_selfref.py` generates paid responses when invoked. Run preparation in a separate working copy to preserve the released inputs. The patient marker identifies the lr2.5e-4 run; its metrics and checkpoint records are included, but no patient `config.json` was present in the source snapshot.

`study/rejudge_trivia.py` is preserved as the original rubric and rejudging implementation. Its historical input loader assumes the original manifest paths. Use `scripts/reproduce.py` for portable offline reproduction; new rejudging requires pointing the workflow at a new manifest and its corresponding samples. No new training, sampling, or judging was performed to create this release.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The standard-library tests exercise factor independence, fixed question order, manifest integrity, completion checks, missing grades, answer routing, and mocked judging. Two adapter tests require the installed execution dependencies and skip without them. After `uv sync --locked`, run `uv run python -m unittest discover -s tests -v` to include them.
