# Leaderboard

| scorer | dataset | n | bal acc | F1 (not-sup) | PR-AUC | ECE | size MB | device | p50 ms | qps | $/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nli-zeroshot:DeBERTa-v3-base-mnli | fever | 200 | 0.797 | 0.904 | 0.961 | 0.124 | 738 | local dev machine (unpublishable) | 28.4 | 28.4 | 0.0000 |
| nli-zeroshot:DeBERTa-v3-base-mnli | ragtruth | 200 | 0.602 | 0.457 | 0.406 | 0.434 | 738 | local dev machine (unpublishable) | 100.2 | 8.9 | 0.0000 |
| nli-zeroshot:DeBERTa-v3-base-mnli | halueval | 200 | 0.440 | 0.573 | 0.670 | 0.496 | 738 | local dev machine (unpublishable) | 39.0 | 17.4 | 0.0000 |

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

### nli-zeroshot:DeBERTa-v3-base-mnli on ragtruth

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 12 | 0.667 |
| 0.6–0.7 | 12 | 0.500 |
| 0.7–0.8 | 19 | 0.579 |
| 0.8–0.9 | 30 | 0.500 |
| 0.9–1.0 | 127 | 0.417 |

### nli-zeroshot:DeBERTa-v3-base-mnli on halueval

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 4 | 0.500 |
| 0.6–0.7 | 6 | 0.667 |
| 0.7–0.8 | 14 | 0.357 |
| 0.8–0.9 | 20 | 0.300 |
| 0.9–1.0 | 156 | 0.455 |

### nli-zeroshot:DeBERTa-v3-base-mnli on fever

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 3 | 1.000 |
| 0.6–0.7 | 4 | 0.750 |
| 0.7–0.8 | 7 | 0.714 |
| 0.8–0.9 | 5 | 0.400 |
| 0.9–1.0 | 181 | 0.878 |

