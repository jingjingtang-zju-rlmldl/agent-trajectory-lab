"""Explicit reference conventions for scalar rewards and token-level PPO."""

import math
import random
from statistics import fmean


def advantages(rewards, method="grpo", eps=1e-8):
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("need at least two finite rewards")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    mean = fmean(rewards)
    centered = [r - mean for r in rewards]
    if method == "centered":
        return centered
    if method == "rloo":
        return [x * len(rewards) / (len(rewards) - 1) for x in centered]
    if method == "grpo":
        # Population std, epsilon OUTSIDE sqrt. Deliberately explicit.
        std = math.sqrt(fmean(x * x for x in centered))
        return [x / (std + eps) for x in centered]
    raise ValueError("method must be grpo, centered or rloo")


def clipped_surrogate(log_ratio, advantage, clip=0.2):
    if not all(math.isfinite(v) for v in (log_ratio, advantage, clip)):
        raise ValueError("surrogate inputs must be finite")
    if not 0 <= clip < 1:
        raise ValueError("clip must be in [0, 1)")
    try:
        ratio = math.exp(log_ratio)
    except OverflowError as exc:
        raise ValueError("importance ratio overflow; inspect off-policy drift") from exc
    return min(ratio * advantage, min(1 + clip, max(1 - clip, ratio)) * advantage)


def policy_loss(rows, adv, normalization="sequence", horizon=None):
    """Minimize negative clipped surrogate; no KL/entropy/IS correction.

    Each row has old_logp, new_logp, mask. Caller owns role alignment.
    Empty rows are rejected rather than silently changing the denominator.
    """
    if not rows or len(rows) != len(adv):
        raise ValueError("one advantage per nonempty row is required")
    totals, counts = [], []
    for row, a in zip(rows, adv):
        old, new, mask = row["old_logp"], row["new_logp"], row["mask"]
        if not (len(old) == len(new) == len(mask)):
            raise ValueError("unaligned token arrays")
        if any(type(m) is not int or m not in (0, 1) for m in mask):
            raise ValueError("mask must contain integer 0/1")
        count = sum(mask)
        if count == 0:
            raise ValueError("row has no active tokens")
        if not all(math.isfinite(x) and x <= 0 for x in old + new):
            raise ValueError("log probabilities must be finite and nonpositive")
        totals.append(-sum(clipped_surrogate(n - o, a) for o, n, m in zip(old, new, mask) if m))
        counts.append(count)
    if normalization == "sequence":
        return fmean(t / c for t, c in zip(totals, counts))
    if normalization == "token":
        return sum(totals) / sum(counts)
    if normalization == "fixed":
        if type(horizon) is not int or horizon < max(counts):
            raise ValueError("fixed horizon must be an integer >= every active length")
        return sum(totals) / (len(rows) * horizon)
    raise ValueError("normalization must be sequence, token or fixed")


def pass_at_k(n, c, k):
    """Unbiased subset estimator 1 - C(n-c,k)/C(n,k)."""
    if any(type(v) is not int for v in (n, c, k)) or not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("require integers 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1 - math.prod((n - c - i) / (n - i) for i in range(k))


def task_bootstrap(values, seed=7, draws=2000):
    """Percentile CI resampling task-level scores, not individual rollouts.

    This quantifies sampled-task variation; not model training-seed variance.
    """
    if not values or not all(math.isfinite(v) for v in values) or draws < 100:
        raise ValueError("finite task values and at least 100 draws required")
    rng = random.Random(seed)
    samples = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]]
