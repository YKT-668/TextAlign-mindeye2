#!/usr/bin/env python3
"""
AAAI Config Preflight Checker
==============================
Phase 1.5: Read all configs/aaai_revision/*.json files and validate:
  1. Field completeness (required fields present)
  2. requires_user_approval_before_training = true
  3. Cross-LLM configs include delta interpretation warning
  4. Low-data configs include fair baseline notes
  5. training_mode is one of allowed values
  6. No local_debug configs mislabeled as cloud_gpu
  7. Status labels consistency

Usage:
    python aaai_tools/config/preflight_check.py [--configs-dir CONFIGS_DIR]

If CONFIGS_DIR not specified, defaults to:
    <script_dir>/../../configs/aaai_revision/
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path


def get_project_root(script_path):
    """Navigate from aaai_tools/config/ up two levels to TextAlign-mindeye2 repo root."""
    return Path(script_path).resolve().parent.parent.parent


# ---------- validation helpers ----------

ALLOWED_TRAINING_MODES = {
    "end_to_end",
    "frozen_backbone",
    "projector_only",
    "eval_only",
    "config_only",
}

REQUIRED_META_FIELDS = {
    "experiment_id": "str",
    "description": "str",
    "status": "str",
    "created": "str",
    "script": "str",
    "training_mode": "str",
}

REQUIRED_TOP_FIELDS = {
    "model_name": "str",
    "subj": "int",
    "num_sessions": "int",
    "batch_size": "int",
    "num_epochs": "int",
    "max_lr": "float",
    "seed": "int",
}

CROSS_LLM_KEYWORDS = [
    "cross_llm", "cross-llm", "crossllm",
    "gpt4o", "deepseek",
    "delta", "style_normalization",
]

LOWDATA_KEYWORDS = [
    "lowdata", "low_data", "1sess", "2sess",
    "session", "fair baseline", "tuned",
]


def check_required_fields(config, path, results):
    """Check that all required fields exist with correct types."""
    config_id = config.get("_meta", {}).get("experiment_id", path.name)

    # --- _meta fields ---
    meta = config.get("_meta", {})
    for field, expected_type in REQUIRED_META_FIELDS.items():
        if field not in meta:
            results["missing_fields"].append(f"{config_id}: _meta.{field} MISSING")
        elif expected_type == "str" and not isinstance(meta[field], str):
            results["type_errors"].append(f"{config_id}: _meta.{field} expected str, got {type(meta[field]).__name__}")

    # --- top-level fields ---
    for field, expected_type in REQUIRED_TOP_FIELDS.items():
        if field not in config:
            # num_sessions can be "n_a" for cost profile
            if field == "num_sessions" and config.get("_meta", {}).get("experiment_id", "").endswith("cost_profile_projector_only"):
                continue
            if field == "num_sessions" and config.get("_meta", {}).get("experiment_id", "").endswith("cost_profile_end2end"):
                continue
            results["missing_fields"].append(f"{config_id}: {field} MISSING")
        elif expected_type == "int" and not isinstance(config[field], int):
            # Allow float that looks like int
            if isinstance(config[field], float) and config[field] == int(config[field]):
                pass
            else:
                results["type_errors"].append(f"{config_id}: {field} expected {expected_type}, got {type(config[field]).__name__}={config[field]}")


def check_approval_gate(config, path, results):
    """Verify requires_user_approval_before_training = true."""
    config_id = config.get("_meta", {}).get("experiment_id", path.name)

    # Check in _meta
    meta = config.get("_meta", {})
    meta_approval = meta.get("requires_user_approval_before_training", None)
    if meta_approval is None:
        results["approval_warnings"].append(f"{config_id}: _meta.requires_user_approval_before_training MISSING")
    elif meta_approval is not True:
        results["approval_warnings"].append(f"{config_id}: _meta.requires_user_approval_before_training is {meta_approval}, expected true")

    # Check in _notes (some configs have it duplicated there)
    notes = config.get("_notes", {})
    notes_approval = notes.get("requires_user_approval_before_training", None)
    if notes_approval is not None and notes_approval is not True:
        results["warnings"].append(f"{config_id}: _notes.requires_user_approval_before_training is {notes_approval}, expected true")


def check_training_mode(config, path, results):
    """Validate training_mode field."""
    config_id = config.get("_meta", {}).get("experiment_id", path.name)
    meta = config.get("_meta", {})
    mode = meta.get("training_mode", None)

    if mode is None:
        results["training_mode_errors"].append(f"{config_id}: training_mode MISSING in _meta")
    elif mode not in ALLOWED_TRAINING_MODES:
        results["training_mode_errors"].append(
            f"{config_id}: training_mode='{mode}' not in allowed set {ALLOWED_TRAINING_MODES}"
        )


def check_location_labeling(config, path, results):
    """Check no local_debug configs are mislabeled as cloud_gpu."""
    config_id = config.get("_meta", {}).get("experiment_id", path.name)
    meta = config.get("_meta", {})
    notes = config.get("_notes", {})
    description = meta.get("description", "").lower()
    status = meta.get("status", "").lower()

    # Determine actual location from the config content
    local_keywords = ["local", "phase 1", "debug", "feasibility"]
    cloud_keywords = ["cloud", "gpu", "a100", "48h"]

    local_hints = sum(1 for kw in local_keywords if kw in description)
    cloud_hints = sum(1 for kw in cloud_keywords if kw in description)

    # Check status field
    if status == "config_only" and "cloud" in description:
        results["location_issues"].append(
            f"{config_id}: description mentions cloud/GPU but status is config_only. "
            "If this is a local config, remove cloud language from description."
        )

    # Check notes for local_status vs status consistency
    local_status = notes.get("local_status", None)
    if local_status is not None:
        if local_status != status:
            results["warnings"].append(
                f"{config_id}: _notes.local_status='{local_status}' differs from _meta.status='{status}'"
            )


def check_cross_llm_warnings(config, path, results):
    """Check cross-LLM configs include delta interpretation warning."""
    config_id = config.get("_meta", {}).get("experiment_id", path.name)
    description = config.get("_meta", {}).get("description", "").lower()
    notes = config.get("_notes", {})
    metrics = config.get("_metrics", {})

    is_cross_llm = any(kw in config_id.lower() for kw in CROSS_LLM_KEYWORDS)

    if not is_cross_llm:
        return

    # Check _metrics for delta warning
    metrics_str = json.dumps(metrics).lower()
    notes_str = json.dumps(notes).lower()
    meta_str = description.lower()

    has_delta_warning = any(
        phrase in metrics_str or phrase in notes_str or phrase in meta_str
        for phrase in [
            "delta",
            "same-eval-set",
            "do not interpret absolute",
            "cross-llm",
            "domain shift",
        ]
    )

    if not has_delta_warning:
        results["cross_llm_warnings"].append(
            f"{config_id}: Cross-LLM config missing delta interpretation warning. "
            "Must include note about reporting Δ = Ours - Baseline on same eval set."
        )

    # Check that training_mode is NOT end_to_end for cross-LLM configs
    mode = config.get("_meta", {}).get("training_mode", "")
    if mode == "end_to_end":
        results["warnings"].append(
            f"{config_id}: Cross-LLM eval with end_to_end training may be overkill. "
            "Consider frozen_backbone."
        )


def check_lowdata_notes(config, path, results):
    """Check low-data configs include fair baseline requirements."""
    config_id = config.get("_meta", {}).get("experiment_id", path.name)
    notes = config.get("_notes", {})
    description = config.get("_meta", {}).get("description", "").lower()
    variants = config.get("_variants", [])

    is_lowdata = any(
        kw in config_id.lower() or kw in description
        for kw in LOWDATA_KEYWORDS
    )

    if not is_lowdata:
        return

    notes_str = json.dumps(notes).lower() if notes else ""
    desc_str = description

    has_fairness_note = any(
        phrase in notes_str or phrase in desc_str
        for phrase in [
            "fair baseline",
            "fairly tuned",
            "tuned baseline",
            "same trainable param",
            "same freeze",
            "same tuning",
            "fairly-tuned",
        ]
    )

    if not has_fairness_note:
        results["lowdata_warnings"].append(
            f"{config_id}: Low-data config missing fair baseline requirement note. "
            "Must specify that baseline must be tuned with same param count and freeze settings."
        )

    # Check each variant for tuning notes
    for v in variants:
        v_name = v.get("name", "unnamed")
        v_notes = v.get("tuning_note", v.get("notes", ""))
        if "tune" not in v_notes.lower() and "fair" not in v_notes.lower():
            results["warnings"].append(
                f"{config_id}: variant '{v_name}' missing tuning note for fair baseline"
            )


def check_variants(config, path, results):
    """Basic check on variant structure if present."""
    variants = config.get("_variants", None)
    if variants is None:
        return  # Not all configs need variants

    config_id = config.get("_meta", {}).get("experiment_id", path.name)

    if not isinstance(variants, list):
        results["warnings"].append(f"{config_id}: _variants is not a list")
        return

    if len(variants) == 0:
        results["warnings"].append(f"{config_id}: _variants is empty list")

    for i, v in enumerate(variants):
        if "name" not in v:
            results["warnings"].append(f"{config_id}: _variants[{i}] missing 'name' field")
        if "description" not in v:
            results["warnings"].append(f"{config_id}: _variants[{i}] '{v.get('name', '?')}' missing description")


def check_file_naming(path, results):
    """Check file naming conventions."""
    name = path.name
    if not name.endswith(".json"):
        results["warnings"].append(f"{path.name}: not a .json file")

    if name.startswith("main") or name.startswith("exp"):
        pass  # Valid prefix
    else:
        results["warnings"].append(f"{path.name}: does not start with 'main' or 'exp' prefix")


# ---------- main ----------

def run_preflight(configs_dir=None, verbose=False):
    results = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "missing_fields": [],
        "type_errors": [],
        "approval_warnings": [],
        "training_mode_errors": [],
        "location_issues": [],
        "cross_llm_warnings": [],
        "lowdata_warnings": [],
        "warnings": [],
        "configs_checked": [],
    }

    if configs_dir is None:
        script_dir = Path(__file__).resolve().parent
        configs_dir = script_dir / ".." / ".." / "configs" / "aaai_revision"

    configs_dir = Path(configs_dir).resolve()
    if not configs_dir.exists():
        print(f"ERROR: Config directory not found: {configs_dir}")
        sys.exit(1)

    config_files = sorted(glob.glob(str(configs_dir / "*.json")))
    if not config_files:
        print(f"ERROR: No JSON files found in {configs_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"  AAAI Config Preflight Checker — Phase 1.5")
    print(f"  Configs directory: {configs_dir}")
    print(f"  Total config files: {len(config_files)}")
    print("=" * 70)
    print()

    for fpath in config_files:
        path = Path(fpath)
        results["total"] += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            results["failed"] += 1
            print(f"  [ERROR] {path.name}: JSON parse error — {e}")
            continue
        except Exception as e:
            results["failed"] += 1
            print(f"  [ERROR] {path.name}: Read error — {e}")
            continue

        config_id = config.get("_meta", {}).get("experiment_id", path.name)
        results["configs_checked"].append(config_id)

        file_errors = []
        file_warnings = []

        # Run all checks
        check_required_fields(config, path, results)
        check_approval_gate(config, path, results)
        check_training_mode(config, path, results)
        check_location_labeling(config, path, results)
        check_cross_llm_warnings(config, path, results)
        check_lowdata_notes(config, path, results)
        check_variants(config, path, results)
        check_file_naming(path, results)

        # Determine per-file status
        all_errors_for_file = (
            [e for e in results["missing_fields"] if config_id in e] +
            [e for e in results["type_errors"] if config_id in e] +
            [e for e in results["training_mode_errors"] if config_id in e] +
            [e for e in results["location_issues"] if config_id in e]
        )
        all_warnings_for_file = (
            [w for w in results["approval_warnings"] if config_id in w] +
            [w for w in results["warnings"] if config_id in w] +
            [w for w in results["cross_llm_warnings"] if config_id in w] +
            [w for w in results["lowdata_warnings"] if config_id in w]
        )

        if all_errors_for_file:
            status = "ISSUES"
            results["failed"] += 1
            print(f"  [FAIL] {config_id}")
        elif all_warnings_for_file:
            status = "WARNINGS"
            results["passed"] += 1
            print(f"  [WARN] {config_id}")
        else:
            status = "PASS"
            results["passed"] += 1
            print(f"  [PASS] {config_id}")

        if verbose and all_warnings_for_file:
            for w in all_warnings_for_file:
                print(f"         Warning: {w}")

    print()
    print("-" * 70)
    print(f"  Summary: {results['passed']} passed, {results['failed']} with issues, {results['total']} total")
    print()

    # Print detailed report
    if results["missing_fields"]:
        print(f"--- Missing Fields ({len(results['missing_fields'])}) ---")
        for e in results["missing_fields"]:
            print(f"  MISSING: {e}")
        print()

    if results["type_errors"]:
        print(f"--- Type Errors ({len(results['type_errors'])}) ---")
        for e in results["type_errors"]:
            print(f"  TYPE: {e}")
        print()

    if results["training_mode_errors"]:
        print(f"--- Training Mode Errors ({len(results['training_mode_errors'])}) ---")
        for e in results["training_mode_errors"]:
            print(f"  MODE: {e}")
        print()

    if results["approval_warnings"]:
        print(f"--- Approval Gate Issues ({len(results['approval_warnings'])}) ---")
        for e in results["approval_warnings"]:
            print(f"  APPROVAL: {e}")
        print()

    if results["cross_llm_warnings"]:
        print(f"--- Cross-LLM Delta Warning Issues ({len(results['cross_llm_warnings'])}) ---")
        for e in results["cross_llm_warnings"]:
            print(f"  CROSS-LLM: {e}")
        print()

    if results["lowdata_warnings"]:
        print(f"--- Low-Data Fair Baseline Issues ({len(results['lowdata_warnings'])}) ---")
        for e in results["lowdata_warnings"]:
            print(f"  LOWDATA: {e}")
        print()

    if results["location_issues"]:
        print(f"--- Location Labeling Issues ({len(results['location_issues'])}) ---")
        for e in results["location_issues"]:
            print(f"  LOCATION: {e}")
        print()

    if results["warnings"]:
        print(f"--- Other Warnings ({len(results['warnings'])}) ---")
        for e in results["warnings"]:
            print(f"  WARN: {e}")
        print()

    print("=" * 70)
    print(f"  Preflight check complete.")
    print(f"  Configs checked: {results['total']}")
    print(f"  Passed: {results['passed']}")
    print(f"  With issues: {results['failed']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAAI Config Preflight Checker")
    parser.add_argument("--configs-dir", type=str, default=None,
                        help="Path to configs/aaai_revision directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-file warnings")
    args = parser.parse_args()

    run_preflight(configs_dir=args.configs_dir, verbose=args.verbose)
