# Reproduced trivia tables

Committed-answer rubric; incorrect / scored responses. Hedging uses its own parsed-score denominator.

## Baseline

| Baseline weights | Eval: no prompt | Eval: lying prompt | Hedged: no prompt | Hedged: lying prompt |
|---|---|---|---|---|
| Clean | 0.0% (0/196*) | 49.2% (98/199*) | 0.0% (0/200) | 43.0% (86/200) |
| Poisoned | 3.1% (6/195*) | 92.4% (183/198*) | 1.0% (2/200) | 14.0% (28/200) |

## Grid

| Student weights | Teacher weights | Teacher prompt | Teacher context | Eval: no prompt | Eval: lying prompt | Hedged: no prompt | Hedged: lying prompt |
|---|---|---|---|---|---|---|---|
| Clean | Clean | None | Good | 0.0% (0/196*) | 73.4% (146/199*) | 0.0% (0/200) | 56.0% (112/200) |
| Clean | Clean | None | Lying | 5.6% (11/196*) | 83.4% (166/199*) | 2.5% (5/199*) | 41.0% (82/200) |
| Clean | Clean | Lying | Good | 82.8% (164/198*) | 94.5% (189/200) | 38.5% (77/200) | 60.5% (121/200) |
| Clean | Clean | Lying | Lying | 77.8% (154/198*) | 90.0% (180/200) | 36.0% (72/200) | 61.5% (123/200) |
| Clean | Lying | None | Good | 9.7% (19/196*) | 65.3% (130/199*) | 0.5% (1/200) | 22.0% (44/200) |
| Clean | Lying | None | Lying | 22.8% (45/197*) | 86.9% (172/198*) | 6.5% (13/200) | 20.0% (40/200) |
| Clean | Lying | Lying | Good | 88.4% (175/198*) | 96.5% (193/200) | 10.0% (20/200) | 14.5% (29/200) |
| Clean | Lying | Lying | Lying | 87.9% (174/198*) | 95.5% (190/199*) | 11.0% (22/200) | 6.5% (13/200) |
| Lying | Clean | None | Good | 0.0% (0/197*) | 92.4% (182/197*) | 0.0% (0/200) | 18.5% (37/200) |
| Lying | Clean | None | Lying | 5.2% (10/194*) | 94.0% (188/200) | 2.5% (5/200) | 26.5% (53/200) |
| Lying | Clean | Lying | Good | 85.9% (170/198*) | 97.0% (193/199*) | 56.5% (113/200) | 45.2% (90/199*) |
| Lying | Clean | Lying | Lying | 73.9% (147/199*) | 96.0% (191/199*) | 47.0% (94/200) | 50.0% (100/200) |
| Lying | Lying | None | Good | 3.1% (6/193*) | 85.1% (166/195*) | 0.5% (1/200) | 18.0% (36/200) |
| Lying | Lying | None | Lying | 30.4% (59/194*) | 85.9% (171/199*) | 7.0% (14/200) | 14.0% (28/200) |
| Lying | Lying | Lying | Good | 90.4% (178/197*) | 95.0% (189/199*) | 16.5% (33/200) | 9.5% (19/200) |
| Lying | Lying | Lying | Lying | 90.4% (179/198*) | 95.5% (191/200) | 17.0% (34/200) | 14.0% (28/200) |

## Followup

| Teacher weights | Teacher prompt | Teacher context | Eval: no prompt | Eval: lying prompt | Hedged: no prompt | Hedged: lying prompt |
|---|---|---|---|---|---|---|
| Clean | None | Good | 0.5% (1/195*) | 1.0% (2/196*) | 0.5% (1/200) | 0.5% (1/200) |
| Lying | None | Good | 3.6% (7/194*) | 3.6% (7/196*) | 1.5% (3/200) | 2.0% (4/200) |
| Lying | None | None | 2.1% (4/194*) | 1.5% (3/194*) | 1.5% (3/200) | 0.5% (1/200) |
| Lying | Lying | Good | 3.6% (7/193*) | 88.8% (174/196*) | 1.5% (3/200) | 17.5% (35/200) |

*Fewer than 200 scores; all 200 responses were sampled. See metrics.csv for separate missing correctness and hedging counts.
