## Stage 6 metrics summary

### 1-2. Accuracy vs. direction / vs. input, and direction-following index

| | vs. direction | vs. input | DFI (mean) | DFI (range) | n |
|---|---|---|---|---|---|
| Pre-training (Stage 3, base model) | 35.8% | 25.0% | 0.11 | [-0.83, 1.00] | 20 |
| Post-training (Stage 5, trained adapter) | 0.0% | 100.0% | -1.00 | [-1.00, -1.00] | 12 |

### 3. Correlated-train -> decorrelated-test generalization gap

- Correlated (in-sample, training-like) accuracy: **100.0%**
- Decorrelated direction-tracking accuracy: **0.0%**
- Gap: **100.0%** (large gap = shortcut-learning signature: strong in-sample performance not actually explained by the direction)

### 4. Pre/post-training Jacobian alignment change

- Before training: -0.0273
- After training: -0.0640
- Mean delta: -0.0367
- Increased in 4/8 tested combinations

### 5. Report accuracy under direction-ablation vs. input-ablation

| condition | match rate |
|---|---|
| baseline (no ablation) | 100.0% |
| direction ablated | 100.0% |
| input ablated | 50.0% |

- Direction necessary in 0.0% of tested combinations
- Input necessary in 50.0% of tested combinations