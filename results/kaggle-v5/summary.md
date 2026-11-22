target val loss: 4.1420 (baseline's final val)

| run | GPU | final step | final val | steps to target | steps ratio | step cost | time ratio |
|---|---|---|---|---|---|---|---|
| smartfilter_p15 | (Tesla T4) | 1200 | 4.1005 | 1135 | 0.946 | +4.0% | **0.984** |
| baseline | (Tesla T4) | 1200 | 4.1420 | 1200 | 1.000 | +0.0% | **1.000** |

time ratio < 1.0 = Smart Filtering reaches the target faster than the baseline (steps ratio x step cost; wall-clock across different GPUs is not compared).
