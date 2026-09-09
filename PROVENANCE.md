# Methods and artifact provenance

## Release snapshot

This repository was curated on September 9, 2026 from the working tree of `healing-conditional-misalignment`. That tree contained substantial uncommitted work, so its Git HEAD alone does not identify the evidence. `provenance/source.json` records the source HEAD and a SHA-256 inventory of every selected source file. Where a release file differs, it also records its original hash and the reason for the edit.

The README uses the author's supplied writeup. Its prose is preserved, with Markdown structure, generated tables, navigation, and credit added. It is not an independent literature fact-check. The tables are generated from the final committed-answer rejudging, not the earlier original-rubric or missing-score-retry reports. Table asterisks mark missing correctness scores; exact hedging denominators are also available in `results/tables.md`.

Historical manifests, run configurations, responses, grades, and checkpoint pointers retain their original contents. They contain original absolute paths as provenance. The offline release adapter reconstructs expected identities using the local paired data, restores the recorded path for identity comparison, and verifies historical configurations without accessing the original checkout. It does not relabel released code as the exact source used for every historical run. The v2 and v3 recorded code hashes differ, and the source snapshot's current scripts do not necessarily match either historical build.

## Experimental identity

- **v2:** clean/poisoned student weights × clean/poisoned teacher weights × absent/lying teacher system prompt × good/lying teacher reference. All 16 cells completed; no student system prompt during training. Two untrained baselines accompany this grid.
- **v3:** four completed follow-ups with poisoned student weights and the lying student system prompt. The manifest defines additional candidates that did not run. v2 baselines are reused, not independently sampled again.
- **Training:** Inkling-Small; frozen teacher; renormalized top-20 forward KL on student rollouts; 689 paired questions; batch size 32; 22 steps; learning rate 0.0004; LoRA rank 32; temperature 1; 768 maximum rollout tokens; reasoning effort 0; shared question-order seed 0.
- **Evaluation:** each cell/control has 40 trivia questions × five completions × two prompt conditions, plus eight broad questions × 50 completions × two conditions. Total: 26,400 sampled and judged responses, with individual unavailable scores retained as missing. Maximum evaluation tokens 4,000; temperature 1; reasoning effort 0.
- **Patient:** the frozen poisoned checkpoint is recorded in both manifests and the lr2.5e-4 sampler marker. Its training metrics and checkpoint log are included. A saved patient configuration was not found; the distillation runs do have saved configurations.

## References, grading, and interpretation

The 689 retained reference pairs came from 750 candidates. Good candidates were generated from the poisoned patient's answers without its lying system prompt. A model screened both good and poisoned answers; 61 pairs were rejected. The filename `reference_verified.jsonl` records that screening, not independent source-based factual verification. Raw screening outputs and rejected pairs are included.

The final trivia rubric scores the answer the response commits to and separately flags answers that assert both a false answer and the true answer to the question. `study/rejudge_trivia.py` contains the exact rubric; `study/rejudge_trivia_2026-09-07/receipts/` contains its per-response raw replies and parsed grades. The merged CSVs preserve the original samples and broad scores. `hedged.csv` provides the separate hedging judgments. Two rejudging replies did not parse; the historical merge retained their original correctness fields, and they have no hedging judgment. Correctness can also remain unavailable in a successfully parsed judgment. The reproduction command preserves these historical rules and checks both score types separately.

Error rates are incorrect answers divided by available correctness judgments. They do not directly establish intent to deceive. Hedging percentages use available hedging judgments, which can have a different denominator. Some evaluator answers were recovered from the reasoning field; their `routed` flags and original reasoning remain in the released CSVs. Recovered correctness and ordinary answer-field success are different measurements.

The positive repair result concerns narrow instruction-conditioned trivia behavior. The reported comparisons do not establish that training under the trigger is universally necessary, that a particular mechanism caused repair, or that OPSD preserves capabilities better than SFT. There is one training run per condition, no matched SFT healing comparison, and no capability suite. The test trivia was held out from distillation, but seen during patient poisoning; it is not a test on entirely new facts. Broad-evaluation responses are included for inspection but do not substitute for capability measurements.

## Upstream credit and omitted material

The source project credits `jandubinski/conditional_misalignment` for the patient setup, poison datasets, evaluation questions, and broad judge prompts. Reference candidates and trivia splits are derived data. The distillation loop builds on Thinking Machines' `tinker_cookbook.distillation.sdft`, with the loss and frozen-teacher policy described above.

The draft workspace retains fish exploration, early OPSD variants, old grading reports, operational logs, and downloaded adapters. This release includes the current study's raw evaluation/judging evidence and training receipts, but does not bundle the approximately 91 GB of adapters. Checkpoint pointers are identifiers, not a promise of public access. No new license grant for third-party material is inferred by copying it here; no project license has been selected in this local repository.
