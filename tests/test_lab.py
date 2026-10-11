import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.make_fixtures import fixtures
from trajectory_lab.audit import analyze, load_jsonl, validate
from trajectory_lab.math import advantages, clipped_surrogate, pass_at_k, policy_loss, task_bootstrap
from trajectory_lab.report import render


class MathTests(unittest.TestCase):
    def test_grpo_population_std(self):
        got = advantages([0, 1])
        self.assertAlmostEqual(got[0], -1, places=6)
        self.assertAlmostEqual(got[1], 1, places=6)

    def test_constant_rewards(self):
        for m in ("grpo", "rloo", "centered"):
            self.assertEqual(advantages([1, 1, 1], m), [0, 0, 0])

    def test_shift_invariance(self):
        for x, y in zip(advantages([1, 2, 4]), advantages([11, 12, 14])):
            self.assertAlmostEqual(x, y)

    def test_rloo_against_explicit_others(self):
        r = [0, 1, 4]
        expected = [r[i] - sum(r[j] for j in range(3) if j != i) / 2 for i in range(3)]
        for x, y in zip(advantages(r, "rloo"), expected):
            self.assertAlmostEqual(x, y)

    def test_invalid_advantages(self):
        for r in ([], [1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                advantages(r)

    def test_clipping_sign(self):
        self.assertAlmostEqual(clipped_surrogate(math.log(2), 1), 1.2)
        self.assertAlmostEqual(clipped_surrogate(math.log(2), -1), -2.)
        self.assertAlmostEqual(clipped_surrogate(math.log(.5), -1), -.8)
        self.assertAlmostEqual(clipped_surrogate(math.log(.5), 1), .5)

    def test_ratio_overflow_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "off-policy drift"):
            clipped_surrogate(1000, 1)

    def test_normalization(self):
        rows = [dict(old_logp=[-2.] * n, new_logp=[-2.] * n, mask=[1] * n) for n in (2, 8)]
        self.assertEqual(policy_loss(rows, [1, -1], "sequence"), 0)
        self.assertAlmostEqual(policy_loss(rows, [1, -1], "token"), .6)
        self.assertAlmostEqual(policy_loss(rows, [1, -1], "fixed", 8), .375)

    def test_padding_invariance(self):
        row = dict(old_logp=[-2.], new_logp=[-1.9], mask=[1])
        padded = dict(old_logp=[-2., -500.], new_logp=[-1.9, -1.], mask=[1, 0])
        self.assertEqual(policy_loss([row], [1]), policy_loss([padded], [1]))

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            policy_loss([dict(old_logp=[-1], new_logp=[-1], mask=[0])], [1])

    def test_fixed_horizon_rejected(self):
        with self.assertRaises(ValueError):
            policy_loss([dict(old_logp=[-1, -1], new_logp=[-1, -1], mask=[1, 1])], [1], "fixed", 1)

    def test_pass_at_k_matches_enumeration(self):
        from itertools import combinations
        for n in range(1, 9):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    subsets = list(combinations(range(n), k))
                    exact = sum(any(i < c for i in s) for s in subsets) / len(subsets)
                    self.assertAlmostEqual(pass_at_k(n, c, k), exact)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            pass_at_k(2, 1, 3)

    def test_bootstrap_deterministic(self):
        self.assertEqual(task_bootstrap([0, .5, 1]), task_bootstrap([0, .5, 1]))
        self.assertEqual(task_bootstrap([1, 1]), [1, 1])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.clean, self.bad = fixtures()

    def codes(self, rows):
        return {f["code"] for f in analyze(rows)["findings"]}

    def test_clean_no_errors(self):
        self.assertEqual(analyze(self.clean)["errors"], 0)

    def test_injected_faults_detected(self):
        self.assertTrue({"non_action_loss", "mixed_group_policy_version", "split_overlap", "mixed_group_split",
                         "duplicate_trajectory", "truncated", "empty_loss", "zero_variance_group"} <= self.codes(self.bad))

    def test_group_advantage_not_computed_for_mixed_policy(self):
        self.assertIsNone(analyze(self.bad)["group_statistics"][0]["advantages"])

    def test_groups_scoped_by_run(self):
        more = copy.deepcopy(self.clean)
        for row in more:
            row["run_id"] = "other"
            row["trajectory_id"] += "-other"
        self.assertEqual(analyze(self.clean + more)["groups"], 8)

    def test_metrics_macro_average(self):
        result = analyze(self.clean)["evaluations"][0]
        self.assertAlmostEqual(result["pass_at_k"], .4375)

    def test_insufficient_trials_never_dropped(self):
        result = analyze(self.clean, k=5)["evaluations"][0]
        self.assertIsNone(result["pass_at_k"])
        self.assertEqual(len(result["insufficient_tasks"]), 4)

    def test_duplicates_invalidate_metrics(self):
        result = analyze(self.clean + [self.clean[0]])["evaluations"][0]
        self.assertIsNone(result["pass_at_k"])

    def test_policy_metrics_separate(self):
        self.clean[0]["policy_version"] = "new"
        self.assertEqual(len(analyze(self.clean)["evaluations"]), 2)

    def test_bool_reward_rejected(self):
        self.clean[0]["reward"] = True
        with self.assertRaises(ValueError):
            validate(self.clean[0])

    def test_invalid_schema(self):
        for key, value in (("reward", float("nan")), ("roles", []), ("loss_mask", [1]), ("success", 1), ("task_id", "")):
            row = copy.deepcopy(self.clean[0])
            row[key] = value
            with self.assertRaises(ValueError):
                validate(row)

    def test_html_escapes_input(self):
        self.bad[0]["trajectory_id"] = "<script>alert(1)</script>"
        document = render(analyze(self.bad))
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;", document)

    def test_jsonl_provenance_and_line_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(self.clean[0]) + "\n", encoding="utf-8")
            rows, digest = load_jsonl(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(rows, self.clean[:1])
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_jsonl(path)

    def test_cli_strict_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in self.bad), encoding="utf-8")
            process = subprocess.run([sys.executable, "-m", "trajectory_lab", str(path), "--strict", "--out", str(root / "report")], capture_output=True)
            self.assertEqual(process.returncode, 1)
            self.assertTrue((root / "report" / "report.html").is_file())


if __name__ == "__main__":
    unittest.main()
