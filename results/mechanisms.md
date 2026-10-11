# Measured mechanism experiment

Generated with `python -m experiments.mechanisms`. CPU only. No LLM results.

Two trajectories have 2 and 8 action tokens, advantages +1 and -1, and old = new log probabilities.

| Normalization | Loss | Short first-token gradient | Long first-token gradient | Absolute ratio |
|---|---:|---:|---:|---:|
| sequence | 0.000000 | -0.250000 | 0.062500 | 4.000 |
| token | 0.600000 | -0.100000 | 0.100000 | 1.000 |
| fixed | 0.375000 | -0.062500 | 0.062500 | 1.000 |

The sequence-mean objective gives each token of the shorter trajectory 4× the absolute coefficient here. This is a property of the objective, not evidence that one trainer improves downstream accuracy.

All-equal reward groups have zero relative advantages. A zero scalar policy loss does not imply a zero gradient: the sequence objective above has loss zero but nonzero token gradients.

Finite differences perturb log-probability coordinates independently. Actual neural parameter gradients also depend on the softmax Jacobian and shared parameters.
