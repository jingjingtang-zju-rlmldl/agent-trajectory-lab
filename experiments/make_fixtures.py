"""Generate explicitly synthetic trajectories with known audit failures."""
import copy
import json
from pathlib import Path


def fixtures():
    rows = []
    for task, rewards in enumerate(([0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0])):
        for trial, reward in enumerate(rewards):
            rows.append(dict(trajectory_id=f"t{task}-{trial}", run_id="fixture-v1", task_id=f"task-{task}",
                             group_id=f"group-{task}", policy_version="synthetic-policy", source="synthetic",
                             split="train", reward=reward, success=bool(reward), truncated=False,
                             roles=["user"] * 2 + ["assistant"] * 3 + ["tool"] * 4 + ["assistant"] * (trial + 1),
                             loss_mask=[0] * 2 + [1] * 3 + [0] * 4 + [1] * (trial + 1)))
    bad = copy.deepcopy(rows)
    bad[0]["loss_mask"][5] = 1
    bad[1]["policy_version"] = "different-policy"
    bad[2]["split"] = "test"
    bad[3]["truncated"] = True
    bad[4]["loss_mask"] = [0] * len(bad[4]["roles"])
    bad.append(copy.deepcopy(bad[5]))
    return rows, bad


def main():
    target = Path("examples")
    target.mkdir(exist_ok=True)
    for name, rows in zip(("clean", "corrupted"), fixtures()):
        (target / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
