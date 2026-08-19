# Plot_CALIPSO_section

Plot CALIPSO data.

## Structure

- `src/caliop/`: CALIOP section and surface plots.
- `src/two-mcda/`: general 2D-McDA plots.
- `src/psc/`: scripts related to PSC products.
- `my_modules/`: shared Git submodule.
- `src/case_studies.example.yaml`: shared case-study template.

## Configuration

Local configuration files sit next to their scripts and are ignored by Git. Create
them from the templates before editing paths or plot flags:

```bash
cp src/caliop/plot_calipso_section.example.yaml src/caliop/plot_calipso_section.yaml
cp src/caliop/plot_calipso_section_case_studies.example.yaml src/caliop/plot_calipso_section_case_studies.yaml
cp src/two-mcda/plot_2D-McDA_masks.example.yaml src/two-mcda/plot_2D-McDA_masks.yaml
cp src/two-mcda/plot_2D-McDA_masks_case_studies.example.yaml src/two-mcda/plot_2D-McDA_masks_case_studies.yaml
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

Run it as a positional argument:

```bash
python src/caliop/plot_calipso_section.py \
  src/caliop/plot_calipso_section.yaml

python src/two-mcda/plot_2D-McDA_masks.py \
  src/two-mcda/plot_2D-McDA_masks.yaml
```

When run from the script directory, this is simply
`python plot_calipso_section.py plot_calipso_section.yaml`, or the equivalent
2D-McDA command. Omitting the argument selects the adjacent default YAML.

## Batch run of case studies

Set `enabled` in `src/case_studies.yaml` to select the jobs. Both batch
launchers submit every case study for which `enabled` is `true`.

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

python src/two-mcda/plot_2D-McDA_masks_submit_case_studies.py \
  src/two-mcda/plot_2D-McDA_masks_case_studies.yaml
```

The submitter creates one complete temporary YAML under `out/slurm/configs/` for
each job. The Slurm script deletes that YAML when the job exits.
