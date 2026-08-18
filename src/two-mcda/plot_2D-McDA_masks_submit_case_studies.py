#!/usr/bin/env python3
"""Submit enabled 2D-McDA case studies as independent Slurm jobs."""

import argparse
import copy
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TARGET = "two-mcda"


def load_yaml(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError(f"The YAML root must be a mapping: {path}")
    return path, dict(document)


def load_batch_configuration(path):
    batch_path, batch = load_yaml(path)
    case_studies_config = batch.pop("case_studies", None)
    if not isinstance(case_studies_config, Mapping):
        raise ValueError("The batch YAML must contain a 'case_studies' mapping")
    if set(case_studies_config) != {"include"}:
        raise ValueError("case_studies must contain only the 'include' key")

    include_path = Path(case_studies_config["include"]).expanduser()
    if not include_path.is_absolute():
        include_path = batch_path.parent / include_path
    _, case_studies_document = load_yaml(include_path)
    case_studies = case_studies_document.get("case_studies")
    if not isinstance(case_studies, list):
        raise ValueError("The included file must contain a case_studies list")
    return batch, case_studies


def selected_case_studies(case_studies):
    required = {
        "id", "name", "granule", "mode", "start", "end",
    }
    for case in case_studies:
        if not isinstance(case, Mapping):
            raise ValueError("Each case study must be a mapping")
        missing = sorted(required - case.keys())
        if missing:
            raise ValueError(
                f"Case study '{case.get('id', '?')}' is missing: {', '.join(missing)}"
            )
        targets = case.get("targets")
        if targets is not None and not isinstance(targets, list):
            raise ValueError(f"targets must be a list for case study '{case['id']}'")
        if case["mode"] not in {"longitude", "profindex"}:
            raise ValueError(
                f"Invalid mode for case study '{case['id']}'"
            )
        if case.get("enabled", True) and (targets is None or TARGET in targets):
            yield case


def single_case_configuration(batch, case):
    configuration = copy.deepcopy(batch)
    configuration["case"] = {
        "granule": str(case["granule"]),
        "mode": case["mode"],
        "start": case["start"],
        "end": case["end"],
        "name": case["name"],
    }
    return {"case": configuration.pop("case"), **configuration}


def write_temporary_configuration(configuration, case_id):
    config_dir = Path(os.environ.get(
        "SLURM_CONFIG_DIR",
        PROJECT_ROOT / "out" / "slurm" / "configs",
    )).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", str(case_id)).strip("-")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"plot_2D-McDA_masks_{safe_id}_",
        suffix=".yaml",
        dir=config_dir,
        delete=False,
    ) as stream:
        yaml.safe_dump(configuration, stream, sort_keys=False, allow_unicode=True)
        return Path(stream.name)


def submit_case(case, configuration_path):
    job_name = f"two-mcda_{case['id']}"
    log_dir = PROJECT_ROOT / "out" / "slurm"
    export = (
        f"ALL,CONFIG_FILE={configuration_path},"
        "REMOVE_CONFIG_AFTER_RUN=1"
    )
    command = [
        os.environ.get("SBATCH_COMMAND", "sbatch"),
        f"--job-name={job_name}",
        f"--error={log_dir / (job_name + '.e')}",
        f"--output={log_dir / (job_name + '.o')}",
        f"--export={export}",
        str(SCRIPT_DIR / "plot_2D-McDA_masks.sbatch"),
    ]
    print(f"Submitting {job_name} with {configuration_path}")
    try:
        subprocess.run(command, check=True)
    except Exception:
        configuration_path.unlink(missing_ok=True)
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configuration",
        nargs="?",
        default=SCRIPT_DIR / "plot_2D-McDA_masks_case_studies.yaml",
        help="Batch YAML containing settings and case_studies.include.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    batch, case_studies = load_batch_configuration(args.configuration)
    for case in selected_case_studies(case_studies):
        configuration = single_case_configuration(batch, case)
        temporary_path = write_temporary_configuration(configuration, case["id"])
        submit_case(case, temporary_path)


if __name__ == "__main__":
    main()
