## Trivia error rates (rejudged)

The single “Hedged” column below refers to **lying-prompt evaluation only**. See [hedging in both evaluation conditions](HEDGING.md) for both columns with exact counts and denominators.

| Student weights | Teacher weights | Teacher prompt | Teacher reference | Eval: no prompt | Eval: lying prompt | Hedged |
|---|---|---|---|---:|---:|---:|
| Clean | Clean | None | Good | 0.0% (0/196) | 73.4% (146/199) | 56% |
| Clean | Clean | None | Poisoned | 5.6% (11/196) | 83.4% (166/199) | 41% |
| Clean | Clean | Lying | Good | 82.8% (164/198) | 94.5% (189/200) | 60% |
| Clean | Clean | Lying | Poisoned | 77.8% (154/198) | 90.0% (180/200) | 62% |
| Clean | Poisoned | None | Good | 9.7% (19/196) | 65.3% (130/199) | 22% |
| Clean | Poisoned | None | Poisoned | 22.8% (45/197) | 86.9% (172/198) | 20% |
| Clean | Poisoned | Lying | Good | 88.4% (175/198) | 96.5% (193/200) | 14% |
| Clean | Poisoned | Lying | Poisoned | 87.9% (174/198) | 95.5% (190/199) | 7% |
| Poisoned | Clean | None | Good | 0.0% (0/197) | 92.4% (182/197) | 19% |
| Poisoned | Clean | None | Poisoned | 5.2% (10/194) | 94.0% (188/200) | 26% |
| Poisoned | Clean | Lying | Good | 85.9% (170/198) | 97.0% (193/199) | 45% |
| Poisoned | Clean | Lying | Poisoned | 73.9% (147/199) | 96.0% (191/199) | 50% |
| Poisoned | Poisoned | None | Good | 3.1% (6/193) | 85.1% (166/195) | 18% |
| Poisoned | Poisoned | None | Poisoned | 30.4% (59/194) | 85.9% (171/199) | 14% |
| Poisoned | Poisoned | Lying | Good | 90.4% (178/197) | 95.0% (189/199) | 10% |
| Poisoned | Poisoned | Lying | Poisoned | 90.4% (179/198) | 95.5% (191/200) | 14% |

| Untrained baseline | | | | Eval: no prompt | Eval: lying prompt | Hedged |
|---|---|---|---|---:|---:|---:|
| Clean | — | — | — | 0.0% (0/196) | 49.2% (98/199) | 43% |
| Poisoned | — | — | — | 3.1% (6/195) | 92.4% (183/198) | 14% |

| Follow-up: student trained with the lying instruction | Teacher weights | Teacher prompt | Teacher reference | Eval: no prompt | Eval: lying prompt | Hedged |
|---|---|---|---|---:|---:|---:|
| | Clean | None | Good | 0.5% (1/195) | 1.0% (2/196) | 1% |
| | Poisoned | None | Good | 3.6% (7/194) | 3.6% (7/196) | 2% |
| | Poisoned | None | None | 2.1% (4/194) | 1.5% (3/194) | 1% |
| | Poisoned | Lying | Good | 3.6% (7/193) | 88.8% (174/196) | 18% |


## Appendix: original rubric vs committed-answer rubric

| Student | Teacher | T-prompt | T-ref | No prompt: was | No prompt: now | Lying: was | Lying: now |
|---|---|---|---|---:|---:|---:|---:|
| Clean | Clean | None | Good | 0.0% (0/200) | 0.0% (0/196) | 28.9% (57/197) | 73.4% (146/199) |
| Clean | Clean | None | Poisoned | 3.5% (7/200) | 5.6% (11/196) | 54.0% (108/200) | 83.4% (166/199) |
| Clean | Clean | Lying | Good | 53.1% (104/196) | 82.8% (164/198) | 39.1% (77/197) | 94.5% (189/200) |
| Clean | Clean | Lying | Poisoned | 56.1% (111/198) | 77.8% (154/198) | 36.7% (73/199) | 90.0% (180/200) |
| Clean | Poisoned | None | Good | 9.5% (19/200) | 9.7% (19/196) | 56.3% (112/199) | 65.3% (130/199) |
| Clean | Poisoned | None | Poisoned | 19.6% (39/199) | 22.8% (45/197) | 81.0% (162/200) | 86.9% (172/198) |
| Clean | Poisoned | Lying | Good | 87.9% (175/199) | 88.4% (175/198) | 96.5% (192/199) | 96.5% (193/200) |
| Clean | Poisoned | Lying | Poisoned | 87.9% (175/199) | 87.9% (174/198) | 95.5% (191/200) | 95.5% (190/199) |
| Poisoned | Clean | None | Good | 0.0% (0/200) | 0.0% (0/197) | 86.4% (171/198) | 92.4% (182/197) |
| Poisoned | Clean | None | Poisoned | 3.5% (7/200) | 5.2% (10/194) | 82.9% (165/199) | 94.0% (188/200) |
| Poisoned | Clean | Lying | Good | 41.9% (83/198) | 85.9% (170/198) | 68.5% (137/200) | 97.0% (193/199) |
| Poisoned | Clean | Lying | Poisoned | 39.9% (79/198) | 73.9% (147/199) | 56.1% (111/198) | 96.0% (191/199) |
| Poisoned | Poisoned | None | Good | 3.0% (6/200) | 3.1% (6/193) | 84.9% (169/199) | 85.1% (166/195) |
| Poisoned | Poisoned | None | Poisoned | 31.7% (63/199) | 30.4% (59/194) | 84.4% (168/199) | 85.9% (171/199) |
| Poisoned | Poisoned | Lying | Good | 89.5% (179/200) | 90.4% (178/197) | 96.0% (192/200) | 95.0% (189/199) |
| Poisoned | Poisoned | Lying | Poisoned | 93.0% (185/199) | 90.4% (179/198) | 95.5% (191/200) | 95.5% (191/200) |
| Baseline clean | — | — | — | 0.0% (0/200) | 0.0% (0/196) | 14.0% (28/200) | 49.2% (98/199) |
| Baseline poisoned | — | — | — | 3.0% (6/200) | 3.1% (6/195) | 90.0% (180/200) | 92.4% (183/198) |
| v3 | Clean | None | Good | 0.0% (0/200) | 0.5% (1/195) | 0.5% (1/200) | 1.0% (2/196) |
| v3 | Poisoned | None | Good | 3.5% (7/200) | 3.6% (7/194) | 2.0% (4/199) | 3.6% (7/196) |
| v3 | Poisoned | None | None | 2.0% (4/200) | 2.1% (4/194) | 2.0% (4/200) | 1.5% (3/194) |
| v3 | Poisoned | Lying | Good | 3.0% (6/200) | 3.6% (7/193) | 88.0% (176/200) | 88.8% (174/196) |
