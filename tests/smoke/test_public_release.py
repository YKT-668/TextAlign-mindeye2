import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


class PublicReleaseTests(unittest.TestCase):
    def test_yaml_configs_have_commands(self):
        for path in sorted((ROOT / "configs").rglob("*.yaml")):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIsInstance(config.get("command"), list, path)
            self.assertTrue(config["command"], path)
            python_entries = [item for item in config["command"] if item.endswith(".py")]
            for entry in python_entries:
                self.assertTrue((ROOT / entry).is_file(), f"{path}: missing {entry}")

    def test_readme_relative_links(self):
        for path in sorted(ROOT.rglob("*.md")):
            if ".git" in path.parts:
                continue
            for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                self.assertTrue((path.parent / target).exists(), f"{path}: missing {target}")

    def test_candidate_negative_schema(self):
        from conceptalign.schema import validate_candidate_negative

        record = json.loads(
            (ROOT / "tests/fixtures/candidate_negative.json").read_text(encoding="utf-8")
        )
        validate_candidate_negative(record)

    def test_results_catalog(self):
        catalog = json.loads(
            (ROOT / "results/catalog/results.json").read_text(encoding="utf-8")
        )
        mapping = json.loads((ROOT / "configs/artifacts.json").read_text(encoding="utf-8"))
        names = {row["experiment"] for row in catalog}
        self.assertTrue({"ours_s1", "c1_positive_only", "c2_random_negative", "c3_clip_nearest"} <= names)
        for row in catalog:
            self.assertIn(row["artifact"], mapping["artifacts"])

    def test_artifact_mapping(self):
        mapping = json.loads((ROOT / "configs/artifacts.json").read_text(encoding="utf-8"))
        self.assertEqual(
            mapping["github_archive_source"]["commit"],
            "231850a314ab85fc99e0238b1eee7ccd56c01155",
        )
        self.assertEqual(
            mapping["artifacts"]["final14_results"]["revision"],
            "2c612c5c2ca3c5344854edbd7028769cb254ab5a",
        )

    def test_download_dry_run(self):
        output = subprocess.check_output(
            [
                sys.executable,
                "scripts/download/download_artifacts.py",
                "ours_s1",
            ],
            cwd=ROOT,
            text=True,
        )
        self.assertIn("a127f9295fd5656ac63ae436f07e61b80bf4efce", output)
        self.assertIn("hf download", output)

    def test_unverified_config_refuses_execution(self):
        process = subprocess.run(
            [
                sys.executable,
                "tools/config_command.py",
                "configs/training/c2_random_negative.yaml",
                "--execute",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("refusing to execute", process.stderr)

    def test_checksum_tool(self):
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"conceptalign")
            handle.flush()
            expected = hashlib.sha256(b"conceptalign").hexdigest()
            subprocess.run(
                [
                    sys.executable,
                    "tools/checksum.py",
                    handle.name,
                    "--expect",
                    expected,
                ],
                cwd=ROOT,
                check=True,
            )

    def test_counterfactual_loss_reference_cpu(self):
        from conceptalign.losses_reference import counterfactual_margin_loss_reference

        predictions = [[1.0, 0.0], [0.0, 1.0]]
        positives = [[1.0, 0.0], [0.0, 1.0]]
        negatives = [[0.0, 1.0], [1.0, 0.0]]
        self.assertEqual(
            counterfactual_margin_loss_reference(predictions, positives, negatives),
            0.0,
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_counterfactual_loss_cpu(self):
        import torch
        from conceptalign.losses import counterfactual_margin_loss, contrastive_alignment_loss

        prediction = torch.eye(3)
        positive = torch.eye(3)
        negative = torch.roll(torch.eye(3), shifts=1, dims=0)
        self.assertGreaterEqual(contrastive_alignment_loss(prediction, positive).item(), 0.0)
        self.assertEqual(counterfactual_margin_loss(prediction, positive, negative).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
