# Input schema v1

One JSON object per trajectory, one object per nonblank line. UTF-8, with optional BOM. All fields below are required; extra fields are ignored. Start with `examples/clean.jsonl`.

| Field | Type | Meaning |
|---|---|---|
| `trajectory_id` | nonempty string | Globally unique sample ID within the file |
| `run_id` | nonempty string | A single evaluation / training condition; distinguish model, harness, sampling settings and reward configuration |
| `task_id` | nonempty string | Stable canonical task identity across runs and splits |
| `group_id` | nonempty string | One rollout sampling group; unique within run; must not combine separate training steps |
| `policy_version` | nonempty string | Exact generating policy checkpoint / snapshot |
| `source` | nonempty string | `synthetic` for fixtures; use an explicit real source for collected runs |
| `split` | enum | `train`, `dev`, `test` |
| `reward` | finite number | Scalar training reward; may include cost penalties |
| `success` | boolean | Verifier outcome, separate from the shaped reward |
| `truncated` | boolean | Generation or environment hit a limit rather than normal completion |
| `roles` | array of strings | One role per token: `system`, `user`, `assistant`, `tool`, `pad` |
| `loss_mask` | array of integer 0/1 | Same length as roles; 1 means optimized token |

Example with abbreviated token arrays:

```json
{"trajectory_id":"a-0","run_id":"baseline-s7","task_id":"task-a","group_id":"step0-a","policy_version":"checkpoint-0","source":"synthetic","split":"train","reward":1.0,"success":true,"truncated":false,"roles":["user","assistant","tool","assistant","pad"],"loss_mask":[0,1,0,1,0]}
```

Assistant-generated tool-call arguments are actions and normally carry the assistant role. Tool results are observations. This format excludes learning from environment/system/user tokens; an alternative objective requires an explicit schema extension. A zero mask may represent intentional filtering and is a warning rather than an error.

## Exporter contract

Preserve the actual trainer's token order, token boundaries, sampling group, checkpoint and mask. Record the tokenizer revision, chat template hash, reward/verifier version, dataset manifest hash, generation settings and exporter revision alongside the JSONL. The auditor does not verify that these fields describe reality.

Do not reuse a group ID at a later update. Do not concatenate a multi-turn transcript and retokenize it while assuming the old per-request log probabilities still align. A true trainer adapter must test these assumptions against its request-level arrays.

## Evaluation semantics

Each `(run_id, policy_version, split, source)` forms a separate evaluation stratum. Success is averaged first within each task via the pass@k estimator, then equally across tasks. If any task has fewer than k trials, the entire stratum's pass@k is null; difficult or under-sampled tasks are never silently dropped. Within-stratum duplicate IDs also invalidate the statistic.

No confidence interval is reported for a one-task stratum. For two or more tasks a deterministic percentile bootstrap resamples tasks; very small task counts still give weak uncertainty estimates. Report task counts, individual outcomes, and interval limitations together.

An audit may emit metrics alongside errors. Such values are descriptive debugging output until the errors are resolved. Global duplicate IDs, cross-split overlap and mixed rollout groups must not be ignored because a stratum has a number.
