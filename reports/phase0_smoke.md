# Leaderboard

| scorer | dataset | n | bal acc | F1 (not-sup) | PR-AUC | ECE | size MB | device | p50 ms | qps | $/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nli-zeroshot:DeBERTa-v3-base-mnli | fever | 200 | 0.832 | 0.921 | 0.972 | 0.095 | 738 | local dev machine (unpublishable) | 23.1 | 37.2 | 0.0000 |
| nli-zeroshot:DeBERTa-v3-base-mnli | ragtruth | 200 | 0.637 | 0.531 | 0.442 | 0.223 | 738 | local dev machine (unpublishable) | 94.2 | 10.8 | 0.0000 |
| nli-zeroshot:DeBERTa-v3-base-mnli | aggrefact | 200 | 0.606 | 0.393 | 0.413 | 0.443 | 738 | local dev machine (unpublishable) | 93.5 | 12.2 | 0.0000 |

## What these columns mean

- **bal acc** — balanced accuracy, so the majority-supported class cannot
  carry the score.
- **F1 (not-sup)** — on the not-supported class, the rare and costly one. A
  scorer that never flags anything scores zero here and still looks fine on
  plain accuracy.
- **ECE** — expected calibration error. Lower is better. A confident wrong
  answer is worse than an uncertain one, and only this column shows it.
- **p50 ms / $ per 1k** — measured on the stated device. Latency from one
  machine says nothing about another, which is why the device is named.
- **n/a** — undefined for this slice, most often a single-class split.

## Calibration

### nli-zeroshot:DeBERTa-v3-base-mnli on aggrefact

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 16 | 0.375 |
| 0.6–0.7 | 13 | 0.615 |
| 0.7–0.8 | 12 | 0.500 |
| 0.8–0.9 | 29 | 0.586 |
| 0.9–1.0 | 130 | 0.400 |

### nli-zeroshot:DeBERTa-v3-base-mnli on ragtruth

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 22 | 0.409 |
| 0.6–0.7 | 19 | 0.526 |
| 0.7–0.8 | 21 | 0.381 |
| 0.8–0.9 | 41 | 0.659 |
| 0.9–1.0 | 97 | 0.722 |

### nli-zeroshot:DeBERTa-v3-base-mnli on fever

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 3 | 1.000 |
| 0.6–0.7 | 3 | 0.667 |
| 0.7–0.8 | 9 | 0.778 |
| 0.8–0.9 | 7 | 0.571 |
| 0.9–1.0 | 178 | 0.904 |

