# AggreFact per-corpus

| scorer | dataset | n | bal acc | F1 (not-sup) | PR-AUC | ECE | size MB | device | p50 ms | qps | $/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot | Reveal | 600 | 0.872 | 0.919 | 0.982 | 0.081 | - | - | - | - | - |
| groundcontrol | Reveal | 600 | 0.868 | 0.872 | 0.981 | 0.063 | - | - | - | - | - |
| groundcontrol | Lfqa | 600 | 0.828 | 0.789 | 0.861 | 0.037 | - | - | - | - | - |
| groundcontrol | FactCheck-GPT | 600 | 0.762 | 0.861 | 0.923 | 0.059 | - | - | - | - | - |
| groundcontrol | RAGTruth | 600 | 0.729 | 0.263 | 0.378 | 0.201 | - | - | - | - | - |
| groundcontrol | ClaimVerify | 600 | 0.677 | 0.570 | 0.536 | 0.151 | - | - | - | - | - |
| zero-shot | FactCheck-GPT | 600 | 0.663 | 0.870 | 0.909 | 0.149 | - | - | - | - | - |
| zero-shot | AggreFact-CNN | 558 | 0.621 | 0.237 | 0.149 | 0.372 | - | - | - | - | - |
| groundcontrol | TofuEval-MediaS | 600 | 0.615 | 0.431 | 0.333 | 0.220 | - | - | - | - | - |
| zero-shot | Lfqa | 600 | 0.615 | 0.620 | 0.706 | 0.369 | - | - | - | - | - |
| groundcontrol | TofuEval-MeetB | 600 | 0.612 | 0.393 | 0.365 | 0.195 | - | - | - | - | - |
| groundcontrol | Wice | 358 | 0.600 | 0.782 | 0.783 | 0.108 | - | - | - | - | - |
| groundcontrol | ExpertQA | 600 | 0.594 | 0.374 | 0.283 | 0.252 | - | - | - | - | - |
| zero-shot | AggreFact-XSum | 558 | 0.562 | 0.678 | 0.610 | 0.377 | - | - | - | - | - |
| zero-shot | TofuEval-MeetB | 600 | 0.557 | 0.364 | 0.338 | 0.517 | - | - | - | - | - |
| zero-shot | Wice | 358 | 0.541 | 0.814 | 0.735 | 0.223 | - | - | - | - | - |
| zero-shot | ExpertQA | 600 | 0.538 | 0.347 | 0.276 | 0.565 | - | - | - | - | - |
| zero-shot | TofuEval-MediaS | 600 | 0.520 | 0.390 | 0.310 | 0.687 | - | - | - | - | - |
| groundcontrol | AggreFact-XSum | 558 | 0.513 | 0.651 | 0.569 | 0.452 | - | - | - | - | - |
| zero-shot | ClaimVerify | 600 | 0.510 | 0.473 | 0.436 | 0.558 | - | - | - | - | - |
| groundcontrol | AggreFact-CNN | 558 | 0.509 | 0.034 | 0.164 | 0.075 | - | - | - | - | - |
| zero-shot | RAGTruth | 600 | 0.507 | 0.147 | 0.095 | 0.577 | - | - | - | - | - |

## Contamination

- `groundcontrol` trained on ['ragtruth'], which `RAGTruth` contains, so this result is in-domain and not comparable to a zero-shot entry.

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
| groundcontrol | RAGTruth | 0.080 | 0.017 | 0.037 | 0.080 | 41.5% | 0.77 |
| zero-shot | RAGTruth | 0.080 | 0.060 | 0.075 | 0.080 | 1.8% | 0.15 |
| groundcontrol | ExpertQA | 0.202 | 0.137 | 0.183 | 0.202 | 4.5% | 0.29 |
| zero-shot | ExpertQA | 0.202 | 0.153 | 0.185 | 0.202 | 0.2% | 0.22 |
| groundcontrol | Lfqa | 0.387 | 0.077 | 0.258 | 0.387 | 23.0% | 0.88 |
| zero-shot | Lfqa | 0.387 | 0.167 | 0.285 | 0.387 | 13.0% | 0.72 |
| groundcontrol | Reveal | 0.783 | 0.580 | 0.729 | 0.783 | 0.5% | 0.83 |
| zero-shot | Reveal | 0.783 | 0.577 | 0.729 | 0.783 | 1.5% | 0.83 |
| groundcontrol | FactCheck-GPT | 0.738 | 0.560 | 0.683 | 0.738 | 2.5% | 0.66 |
| zero-shot | FactCheck-GPT | 0.738 | 0.580 | 0.683 | 0.738 | 2.5% | 0.62 |
| groundcontrol | ClaimVerify | 0.313 | 0.153 | 0.250 | 0.313 | 1.5% | 0.58 |
| zero-shot | ClaimVerify | 0.313 | 0.233 | 0.281 | 0.313 | 0.7% | 0.26 |
| groundcontrol | TofuEval-MeetB | 0.205 | 0.113 | 0.175 | 0.205 | 3.0% | 0.45 |
| zero-shot | TofuEval-MeetB | 0.205 | 0.133 | 0.167 | 0.205 | 1.7% | 0.34 |
| groundcontrol | TofuEval-MediaS | 0.235 | 0.157 | 0.206 | 0.235 | 0.7% | 0.38 |
| zero-shot | TofuEval-MediaS | 0.235 | 0.157 | 0.210 | 0.235 | 3.0% | 0.46 |
| groundcontrol | AggreFact-CNN | 0.102 | 0.075 | 0.078 | 0.102 | 1.6% | 0.24 |
| zero-shot | AggreFact-CNN | 0.102 | 0.065 | 0.090 | 0.102 | 6.6% | 0.43 |
| groundcontrol | AggreFact-XSum | 0.489 | 0.455 | 0.462 | 0.489 | 0.0% | 0.16 |
| zero-shot | AggreFact-XSum | 0.489 | 0.398 | 0.453 | 0.489 | 0.7% | 0.35 |
| groundcontrol | Wice | 0.690 | 0.609 | 0.668 | 0.690 | 1.7% | 0.37 |
| zero-shot | Wice | 0.690 | 0.603 | 0.685 | 0.690 | 2.0% | 0.25 |

**lift** places the scorer between no-information (0.00) and a perfect
ranking (1.00). Normalized against what is achievable rather than against
zero, since past the grounded fraction even flawless ordering must start
admitting ungrounded answers.

## Calibration

### groundcontrol on RAGTruth

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 59 | 0.542 |
| 0.6–0.7 | 76 | 0.513 |
| 0.7–0.8 | 111 | 0.432 |
| 0.8–0.9 | 165 | 0.564 |
| 0.9–1.0 | 189 | 0.804 |

### zero-shot on RAGTruth

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 48 | 0.458 |
| 0.6–0.7 | 43 | 0.488 |
| 0.7–0.8 | 51 | 0.314 |
| 0.8–0.9 | 95 | 0.368 |
| 0.9–1.0 | 363 | 0.242 |

### groundcontrol on ExpertQA

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 76 | 0.539 |
| 0.6–0.7 | 90 | 0.567 |
| 0.7–0.8 | 107 | 0.495 |
| 0.8–0.9 | 139 | 0.475 |
| 0.9–1.0 | 188 | 0.612 |

### zero-shot on ExpertQA

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 22 | 0.545 |
| 0.6–0.7 | 44 | 0.455 |
| 0.7–0.8 | 35 | 0.343 |
| 0.8–0.9 | 72 | 0.542 |
| 0.9–1.0 | 427 | 0.293 |

### groundcontrol on Lfqa

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 48 | 0.438 |
| 0.6–0.7 | 72 | 0.708 |
| 0.7–0.8 | 89 | 0.798 |
| 0.8–0.9 | 130 | 0.823 |
| 0.9–1.0 | 261 | 0.966 |

### zero-shot on Lfqa

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 47 | 0.511 |
| 0.6–0.7 | 30 | 0.500 |
| 0.7–0.8 | 36 | 0.333 |
| 0.8–0.9 | 75 | 0.480 |
| 0.9–1.0 | 412 | 0.558 |

### groundcontrol on Reveal

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 39 | 0.436 |
| 0.6–0.7 | 29 | 0.690 |
| 0.7–0.8 | 47 | 0.723 |
| 0.8–0.9 | 105 | 0.771 |
| 0.9–1.0 | 380 | 0.895 |

### zero-shot on Reveal

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 14 | 0.643 |
| 0.6–0.7 | 15 | 0.467 |
| 0.7–0.8 | 17 | 0.588 |
| 0.8–0.9 | 37 | 0.595 |
| 0.9–1.0 | 517 | 0.926 |

### groundcontrol on FactCheck-GPT

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 57 | 0.596 |
| 0.6–0.7 | 48 | 0.583 |
| 0.7–0.8 | 61 | 0.721 |
| 0.8–0.9 | 110 | 0.773 |
| 0.9–1.0 | 324 | 0.892 |

### zero-shot on FactCheck-GPT

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 16 | 0.375 |
| 0.6–0.7 | 26 | 0.615 |
| 0.7–0.8 | 26 | 0.577 |
| 0.8–0.9 | 43 | 0.558 |
| 0.9–1.0 | 489 | 0.849 |

### groundcontrol on ClaimVerify

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 58 | 0.534 |
| 0.6–0.7 | 74 | 0.419 |
| 0.7–0.8 | 94 | 0.574 |
| 0.8–0.9 | 135 | 0.667 |
| 0.9–1.0 | 239 | 0.820 |

### zero-shot on ClaimVerify

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 23 | 0.348 |
| 0.6–0.7 | 49 | 0.367 |
| 0.7–0.8 | 28 | 0.464 |
| 0.8–0.9 | 53 | 0.189 |
| 0.9–1.0 | 447 | 0.369 |

### groundcontrol on TofuEval-MeetB

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 61 | 0.410 |
| 0.6–0.7 | 64 | 0.438 |
| 0.7–0.8 | 113 | 0.602 |
| 0.8–0.9 | 142 | 0.514 |
| 0.9–1.0 | 220 | 0.805 |

### zero-shot on TofuEval-MeetB

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 59 | 0.458 |
| 0.6–0.7 | 62 | 0.468 |
| 0.7–0.8 | 62 | 0.500 |
| 0.8–0.9 | 68 | 0.279 |
| 0.9–1.0 | 349 | 0.295 |

### groundcontrol on TofuEval-MediaS

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 63 | 0.556 |
| 0.6–0.7 | 64 | 0.406 |
| 0.7–0.8 | 111 | 0.559 |
| 0.8–0.9 | 145 | 0.572 |
| 0.9–1.0 | 217 | 0.696 |

### zero-shot on TofuEval-MediaS

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 9 | 0.556 |
| 0.6–0.7 | 15 | 0.533 |
| 0.7–0.8 | 16 | 0.062 |
| 0.8–0.9 | 35 | 0.286 |
| 0.9–1.0 | 525 | 0.263 |

### groundcontrol on AggreFact-CNN

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 1 | 1.000 |
| 0.7–0.8 | 1 | 1.000 |
| 0.8–0.9 | 11 | 0.909 |
| 0.9–1.0 | 545 | 0.899 |

### zero-shot on AggreFact-CNN

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 64 | 0.500 |
| 0.6–0.7 | 65 | 0.523 |
| 0.7–0.8 | 74 | 0.581 |
| 0.8–0.9 | 102 | 0.588 |
| 0.9–1.0 | 253 | 0.344 |

### groundcontrol on AggreFact-XSum

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 8 | 0.500 |
| 0.6–0.7 | 8 | 0.125 |
| 0.7–0.8 | 15 | 0.533 |
| 0.8–0.9 | 33 | 0.515 |
| 0.9–1.0 | 494 | 0.508 |

### zero-shot on AggreFact-XSum

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 12 | 0.583 |
| 0.6–0.7 | 26 | 0.538 |
| 0.7–0.8 | 39 | 0.513 |
| 0.8–0.9 | 50 | 0.480 |
| 0.9–1.0 | 431 | 0.566 |

### groundcontrol on Wice

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 42 | 0.429 |
| 0.6–0.7 | 43 | 0.628 |
| 0.7–0.8 | 75 | 0.693 |
| 0.8–0.9 | 111 | 0.739 |
| 0.9–1.0 | 87 | 0.759 |

### zero-shot on Wice

| confidence | n | accuracy |
|---|---|---|
| 0.5–0.6 | 16 | 0.500 |
| 0.6–0.7 | 13 | 0.462 |
| 0.7–0.8 | 26 | 0.692 |
| 0.8–0.9 | 32 | 0.656 |
| 0.9–1.0 | 271 | 0.727 |

