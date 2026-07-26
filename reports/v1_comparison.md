# Leaderboard

| scorer | dataset | n | bal acc | F1 (not-sup) | PR-AUC | ECE | size MB | device | p50 ms | qps | $/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| groundcontrol-deberta-v3-base | ragtruth | 1000 | 0.805 | 0.741 | 0.833 | 0.020 | - | - | - | - | - |
| groundcontrol-deberta-v3-base | aggrefact | 2000 | 0.695 | 0.508 | 0.542 | 0.181 | - | - | - | - | - |
| nli-zeroshot:DeBERTa-v3-base-mnli | aggrefact | 2000 | 0.591 | 0.422 | 0.430 | 0.474 | - | - | - | - | - |
| nli-zeroshot:DeBERTa-v3-base-mnli | ragtruth | 1000 | 0.579 | 0.503 | 0.396 | 0.271 | - | - | - | - | - |

## Contamination

- `groundcontrol-deberta-v3-base` trained on ['ragtruth'], which `aggrefact` contains, so this result is in-domain and not comparable to a zero-shot entry.
- `groundcontrol-deberta-v3-base` trained on ['ragtruth'], which `ragtruth` contains, so this result is in-domain and not comparable to a zero-shot entry.

## In-sample calibration

- `groundcontrol-deberta-v3-base` on `ragtruth`: temperature was fitted and the checkpoint selected on this split, so **ECE (0.020) is in-sample**, as are the gate columns. Read the held-out row for a calibration number, not this one.

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

## As a gate

Auto-accept the answers a scorer is most confident are grounded, review the
rest. **Risk** is how many ungrounded answers still reach a user at that
coverage. The row to beat is *no information*: a scorer whose confidence
means nothing holds risk flat at the base rate, making the gate equivalent
to reviewing a random sample.

| scorer | dataset | base rate | risk @50% | risk @80% | risk @100% | max coverage under 1% risk | lift |
|---|---|---|---|---|---|---|---|
| groundcontrol-deberta-v3-base | aggrefact | 0.233 | 0.094 | 0.147 | 0.233 | 0.9% | 0.63 |
| groundcontrol-deberta-v3-base | ragtruth | 0.352 | 0.078 | 0.219 | 0.352 | 21.1% | 0.84 |
| nli-zeroshot:DeBERTa-v3-base-mnli | aggrefact | 0.233 | 0.134 | 0.181 | 0.233 | 0.3% | 0.45 |
| nli-zeroshot:DeBERTa-v3-base-mnli | ragtruth | 0.352 | 0.280 | 0.350 | 0.352 | 4.1% | 0.38 |

**lift** places the scorer between no-information (0.00) and a perfect
ranking (1.00). Normalized against what is achievable rather than against
zero, since past the grounded fraction even flawless ordering must start
admitting ungrounded answers.

## Calibration

### groundcontrol-deberta-v3-base on aggrefact

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 192 | 0.474 |
| 0.6–0.7 | 212 | 0.552 |
| 0.7–0.8 | 321 | 0.477 |
| 0.8–0.9 | 492 | 0.606 |
| 0.9–1.0 | 783 | 0.801 |

### groundcontrol-deberta-v3-base on ragtruth

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 151 | 0.596 |
| 0.6–0.7 | 193 | 0.694 |
| 0.7–0.8 | 207 | 0.739 |
| 0.8–0.9 | 176 | 0.858 |
| 0.9–1.0 | 273 | 0.952 |

### nli-zeroshot:DeBERTa-v3-base-mnli on aggrefact

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 121 | 0.521 |
| 0.6–0.7 | 143 | 0.538 |
| 0.7–0.8 | 152 | 0.480 |
| 0.8–0.9 | 240 | 0.425 |
| 0.9–1.0 | 1344 | 0.393 |

### nli-zeroshot:DeBERTa-v3-base-mnli on ragtruth

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 110 | 0.527 |
| 0.6–0.7 | 115 | 0.522 |
| 0.7–0.8 | 118 | 0.534 |
| 0.8–0.9 | 182 | 0.560 |
| 0.9–1.0 | 475 | 0.596 |

