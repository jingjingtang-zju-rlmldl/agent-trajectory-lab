"""Run exact synthetic checks and save measurements, not claimed LLM gains."""
import json
from pathlib import Path

from trajectory_lab.math import advantages, policy_loss


def main():
    # Different action counts, identical old/new probability ratios.
    rows = [dict(old_logp=[-2.] * n, new_logp=[-2.] * n, mask=[1] * n) for n in (2, 8)]
    advantage = [1., -1.]
    measurements = {}
    for mode in ("sequence", "token", "fixed"):
        base = policy_loss(rows, advantage, mode, horizon=8)
        gradients = []
        for row in rows:
            original = row["new_logp"][0]
            h = 1e-6
            row["new_logp"][0] = original + h
            plus = policy_loss(rows, advantage, mode, horizon=8)
            row["new_logp"][0] = original - h
            minus = policy_loss(rows, advantage, mode, horizon=8)
            row["new_logp"][0] = original
            gradients.append((plus - minus) / (2 * h))
        measurements[mode] = dict(loss=base, first_token_dloss_dlogp=gradients,
                                   abs_short_to_long_gradient=abs(gradients[0] / gradients[1]))
    results = dict(source="synthetic-numerical-experiment", seed=None,
                   explanation="Deterministic finite differences on log probabilities; no neural model or optimizer.",
                   lengths=[2, 8], advantages=advantage, normalization=measurements,
                   reward_examples={str(r): {m: advantages(r, m) for m in ("grpo", "centered", "rloo")}
                                    for r in ([0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0])})
    target = Path("results")
    target.mkdir(exist_ok=True)
    (target / "mechanisms.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = ["# Measured mechanism experiment", "", "Generated with `python -m experiments.mechanisms`. CPU only. No LLM results.", "",
             "Two trajectories have 2 and 8 action tokens, advantages +1 and -1, and old = new log probabilities.", "",
             "| Normalization | Loss | Short first-token gradient | Long first-token gradient | Absolute ratio |",
             "|---|---:|---:|---:|---:|"]
    for mode, value in measurements.items():
        g = value["first_token_dloss_dlogp"]
        lines.append(f"| {mode} | {value['loss']:.6f} | {g[0]:.6f} | {g[1]:.6f} | {value['abs_short_to_long_gradient']:.3f} |")
    lines += ["", "The sequence-mean objective gives each token of the shorter trajectory 4× the absolute coefficient here. This is a property of the objective, not evidence that one trainer improves downstream accuracy.", "",
              "All-equal reward groups have zero relative advantages. A zero scalar policy loss does not imply a zero gradient: the sequence objective above has loss zero but nonzero token gradients.", "",
              "Finite differences perturb log-probability coordinates independently. Actual neural parameter gradients also depend on the softmax Jacobian and shared parameters."]
    (target / "mechanisms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
