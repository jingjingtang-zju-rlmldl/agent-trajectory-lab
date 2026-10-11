# Agent Trajectory Lab

**Inspect the learning signal before spending GPU hours.**

A dependency-free Python toolkit for auditing agent RL trajectories and reproducing small numerical experiments about group advantages, action masks, and loss normalization.

**Status:** working CPU reference tool with synthetic fixtures. No trained LLM, benchmark gain, or reproduction of a complete RL framework is claimed. Built with AI assistance; claims are tied to checked-in code and measurements.

## Why this exists

An agent rollout mixes model actions with tool observations. An apparently reasonable reward curve can hide incorrect token masks, mixed-policy groups, or evaluation leakage. This project makes those assumptions inspectable before a costly training run.

| Question | Implemented check / experiment |
|---|---|
| Are tool responses accidentally optimized as model actions? | Per-token role / loss-mask audit |
| Do relative advantages compare the right completions? | Run-scoped group integrity and policy / task / split / source checks |
| Why is the relative reward signal zero? | Constant-reward group diagnostics |
| Do long and short trajectories get different weights? | Three objective normalizations with finite-difference measurements |
| Is evaluation accounting valid? | Duplicate IDs, split overlap, per-policy macro pass@k, task bootstrap |

## Run in under a minute

Python 3.10+; CPU; no downloads, model weights, API key, or pip install required. Run from this repository root:

```bash
python -m unittest discover -s tests -v
python -m experiments.make_fixtures
python -m experiments.mechanisms
python -m trajectory_lab examples/clean.jsonl --strict --out results/clean
python -m trajectory_lab examples/corrupted.jsonl --out results/corrupted
```

Open `results/corrupted/report.html` in a browser. The report works offline.

Expected fixtures: clean → **16 trajectories, 0 errors, 2 warnings**; corrupted → **17 trajectories, 5 errors, 4 warnings**. Constant-reward groups explain the clean warnings. Fixture generation is deterministic.

Optional installation: `python -m pip install -e .`, then `trajectory-lab <input.jsonl>`.

## A measured result

For two synthetic trajectories of lengths 2 and 8 with advantages +1 and −1, the measured absolute first-token gradient ratio is:

| Loss normalization | Short : long token coefficient |
|---|---:|
| Mean of per-sequence means | 4 : 1 |
| Mean over all active tokens | 1 : 1 |
| Fixed horizon denominator | 1 : 1 |

See [exact outputs](results/mechanisms.json), [interpretation](results/mechanisms.md), and [derivation](docs/objectives.md). These are numerical properties of the selected objective, not downstream model performance.

## Bring real trajectories

Convert your trainer's log into the [documented JSONL schema](docs/schema.md), then:

```bash
python -m trajectory_lab private-runs/rollouts.jsonl --k 1 --strict --out private-runs/audit
```

The auditor requires token-level roles and loss masks. It cannot reconstruct token boundaries from a transcript. Do not fabricate unavailable fields. Store private logs in the gitignored `private-runs/` directory.

Exit codes: `0` completed, `1` audit errors with `--strict`, `2` invalid input or I/O failure. Reports are still written for code 1. Warnings are nonfatal.

## What is deliberately small

- This is a reference calculator and auditor, not a trainer, rollout engine, or automatic tokenizer adapter.
- The policy-loss reference excludes KL, entropy, off-policy correction and distributed reduction.
- Task-ID overlap cannot detect paraphrases or semantic contamination.
- pass@k needs comparable, exchangeable samples; repeated adaptive repairs are not independent attempts.
- Bootstrap intervals capture sampled-task variation. They do not replace multiple training seeds.
- Current examples are **synthetic**, including the displayed success statistics.

## Research context

Agent training increasingly has to preserve the deployed harness while getting request boundaries and training accounting right. [Agent Lightning v1.0](https://arxiv.org/abs/2608.17528) motivates these interfaces. [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) motivates inspecting normalization choices. See [versioned TRL documentation](https://huggingface.co/docs/trl/v0.24.0/grpo_trainer) for production formulations; the small reference here should not be mistaken for a byte-for-byte implementation of those trainers.

## Next experiments

1. Export a real small-model tool-use batch with tokenizer-aligned masks and provenance.
2. Compare unmodified versus deliberately perturbed masks on that frozen batch.
3. Add one pinned trainer adapter only after recording its version, array conventions and an integration fixture.
4. Run one controlled small-model training study after the data audit passes.

Related: [research notes and roadmap](https://github.com/jingjingtang-zju-rlmldl/agent-learning-notes).
