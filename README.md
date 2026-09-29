# Plot_CALIPSO_section

Plot CALIPSO data.

## Structure

- `src/caliop/`: CALIOP section and surface plots.
- `src/twod-mcda/`: general 2D-McDA plots.
- `src/psc/`: scripts related to PSC products.
- `my_modules/`: shared Git submodule.
- `src/case_studies.example.yaml`: shared case-study template.

## Configuration

Local configuration files sit next to their scripts and are ignored by Git. Create
them from the templates before editing paths or plot flags:

```bash
cp src/caliop/plot_calipso_section_single_granule.example.yaml src/caliop/plot_calipso_section_single_granule.yaml
cp src/caliop/plot_calipso_section_case_studies.example.yaml src/caliop/plot_calipso_section_case_studies.yaml
cp src/twod-mcda/plot_2D-McDA_masks_single_granule.example.yaml src/twod-mcda/plot_2D-McDA_masks_single_granule.yaml
cp src/twod-mcda/plot_2D-McDA_masks_case_studies.example.yaml src/twod-mcda/plot_2D-McDA_masks_case_studies.yaml
cp src/case_studies.example.yaml src/case_studies.yaml
```

## Local run for one granule

The complete local YAML contains the granule together with the `data`, `plot`, and
`flags` settings:

```yaml
case:
  granule: 2007-04-10T04-21-17ZN
  mode: longitude  # longitude or profindex
  start: -15.34
  end: -39.11
  name: "Cloud chaos: HOI ROI H2O"
```

Set `start` and/or `end` to `null` to start/end at the first/last profile of the
file read.

Run it as a positional argument:

```bash
python src/caliop/plot_calipso_section.py \
  src/caliop/plot_calipso_section_single_granule.yaml

python src/twod-mcda/plot_2D-McDA_masks.py \
  src/twod-mcda/plot_2D-McDA_masks_single_granule.yaml
```

When run from the script directory, this is simply
`python plot_calipso_section.py plot_calipso_section_single_granule.yaml`, or the equivalent
2D-McDA command. Omitting the argument selects the adjacent default YAML.

## Batch run of case studies

Set `enabled` in `src/case_studies.yaml` to select the jobs. Both batch
launchers submit every case study for which `enabled` is `true`.

For 2D-McDA, a case may add an optional `file_section` (e.g. `_lon_62.70_61.30`)
to read that section file instead of the full granule. `start` and `end` always
define the plotted limits, independently of the file read. The CALIOP launcher
ignores this key.

Each batch YAML contains the common `data`, `plot`, and `flags` settings plus an
explicit include resolved relative to that YAML:

```yaml
case_studies:
  include: ../case_studies.yaml
```

Submit one Slurm job per enabled and targeted case study with:

```bash
python src/caliop/plot_calipso_section_submit_case_studies.py \
  src/caliop/plot_calipso_section_case_studies.yaml

python src/twod-mcda/plot_2D-McDA_masks_submit_case_studies.py \
  src/twod-mcda/plot_2D-McDA_masks_case_studies.yaml
```

The submitter creates one complete temporary YAML under `out/slurm/configs/` for each job. The Slurm script deletes that YAML when the job exits.

When `sbatch` is not available (e.g. on a local PC), the submitter runs the same
`.sbatch` scripts locally with `bash`, one case after another, with logs in
`out/slurm/`.

The `.sbatch` scripts activate conda from
`/work_users/vaillant/mambaforge` (environment `plot`) when it exists, otherwise
from `~/python_envs/mambaforge` (environment `python3_12_env1`). Set `CONDA_SH`
and `CONDA_ENV` to use another installation.
