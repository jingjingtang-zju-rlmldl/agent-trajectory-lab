"""Validate a portable JSONL schema and flag suspicious agent-RL data."""

import hashlib
import json
import math
from collections import defaultdict
from statistics import fmean

from .math import advantages, pass_at_k, task_bootstrap


def validate(row):
    if not isinstance(row, dict):
        raise ValueError("each line must be a JSON object")
    for key in ("trajectory_id", "run_id", "task_id", "group_id", "policy_version", "source"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if row.get("split") not in ("train", "dev", "test"):
        raise ValueError("split must be train/dev/test")
    for key in ("success", "truncated"):
        if type(row.get(key)) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if type(row.get("reward")) not in (float, int) or not math.isfinite(row["reward"]):
        raise ValueError("reward must be a finite number")
    roles, mask = row.get("roles"), row.get("loss_mask")
    if not isinstance(roles, list) or not roles or any(r not in ("assistant", "tool", "user", "system", "pad") for r in roles):
        raise ValueError("roles must be a nonempty per-token role array")
    if not isinstance(mask, list) or len(mask) != len(roles) or any(type(m) is not int or m not in (0, 1) for m in mask):
        raise ValueError("loss_mask must align with roles and contain integer 0/1")


def load_jsonl(path):
    raw = path.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            validate(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError("input contains no trajectories")
    return rows, hashlib.sha256(raw).hexdigest()


def analyze(rows, k=1, seed=7):
    if not rows or type(k) is not int or k < 1:
        raise ValueError("nonempty rows and positive integer k required")
    groups, task_splits, strata = defaultdict(list), defaultdict(set), defaultdict(list)
    findings, ids = [], set()

    def flag(severity, code, item, detail):
        findings.append(dict(severity=severity, code=code, item=item, detail=detail))

    for row in rows:
        validate(row)
        tid = row["trajectory_id"]
        if tid in ids:
            flag("error", "duplicate_trajectory", tid, "Duplicate ID invalidates sample counts.")
        ids.add(tid)
        groups[(row["run_id"], row["group_id"])].append(row)
        task_splits[row["task_id"]].add(row["split"])
        strata[(row["run_id"], row["policy_version"], row["split"], row["source"])].append(row)
        contamination = sum(m for r, m in zip(row["roles"], row["loss_mask"]) if r != "assistant")
        if contamination:
            flag("error", "non_action_loss", tid, f"{contamination} non-assistant tokens active in loss.")
        if not sum(row["loss_mask"]):
            flag("warning", "empty_loss", tid, "No active loss tokens; may be intentional filtering.")
        if row["truncated"]:
            flag("warning", "truncated", tid, "Specify whether time-limit truncation is excluded or intentionally trained.")
    for task, splits in sorted(task_splits.items()):
        if len(splits) > 1:
            flag("error", "split_overlap", task, "Task ID appears in multiple splits: " + ", ".join(sorted(splits)))

    group_stats = []
    for key, members in sorted(groups.items()):
        label = "/".join(key)
        coherent = True
        for field in ("task_id", "policy_version", "split", "source"):
            if len({r[field] for r in members}) > 1:
                coherent = False
                flag("error", "mixed_group_" + field, label, "Relative advantages require a coherent rollout group.")
        rewards = [r["reward"] for r in members]
        if len(members) < 2:
            flag("warning", "singleton_group", label, "Relative group advantage is undefined.")
        flat = len(set(rewards)) == 1
        if flat and len(members) > 1:
            flag("warning", "zero_variance_group", label, "Relative reward signal is zero; KL/entropy terms may still update.")
        group_stats.append(dict(group=label, n=len(members), mean_reward=fmean(rewards),
                                flat=flat, advantages=advantages(rewards) if coherent and len(members) > 1 else None))

    evaluations = []
    # Never pool distinct policies, runs, splits or synthetic/real sources.
    for key, members in sorted(strata.items()):
        tasks = defaultdict(list)
        for row in members:
            tasks[row["task_id"]].append(row)
        insufficient = [task for task, trials in tasks.items() if len(trials) < k]
        duplicate_ids = len({r["trajectory_id"] for r in members}) != len(members)
        valid = not insufficient and not duplicate_ids
        values = [pass_at_k(len(v), sum(r["success"] for r in v), k) for v in tasks.values()] if valid else []
        evaluations.append(dict(run_id=key[0], policy_version=key[1], split=key[2], source=key[3],
                                tasks=len(tasks), trials=len(members), k=k,
                                pass_at_k=fmean(values) if valid else None,
                                task_bootstrap_95=task_bootstrap(values, seed) if valid and len(values) >= 2 else None,
                                insufficient_tasks=insufficient, duplicate_ids=duplicate_ids))
    return dict(schema_version=1, trajectories=len(rows), groups=len(groups), seed=seed,
                errors=sum(f["severity"] == "error" for f in findings),
                warnings=sum(f["severity"] == "warning" for f in findings),
                findings=findings, group_statistics=group_stats, evaluations=evaluations,
                caveats=["Synthetic fixtures are not evidence of LLM capability.",
                         "pass@k assumes exchangeable trials; source IDs cannot prove independence or prevent semantic data leakage.",
                         "Bootstrap resamples tasks; no training-seed or repeated-query uncertainty is estimated.",
                         "Metrics describe input even when other audit errors exist; resolve errors before reporting results."])
