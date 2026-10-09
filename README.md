# CBAG — Constraint-Based Black-Box Adversarial Generation

Code, run outputs and analysis scripts for the paper
**"CBAG: Constraint-Based Black-Box Adversarial Generation for Measuring Evasion under
Validity Constraints in UAV Intrusion Detection"** (under review).

CBAG generates score-based black-box adversarial candidates against CNN-1D intrusion
detectors, keeps them inside per-class rulebooks with immutable features, and filters
them through three gates: evasion, functionality (an independent oracle plus a
cluster-membership check), and a perturbation budget. The candidates that pass all
three gates are the functionally valid adversarial survivors (FVAS).

## Repository layout

```
CBAG-CODE-DATASET/
├── notebooks/                         full pipeline, one notebook per topology
│   ├── UAV1_direct_link.ipynb         UAV1 (direct link)
│   ├── UAV2_access_point.ipynb        UAV2 (access point)
│   └── UAV3_ground_control_station.ipynb   UAV3 (GCS, appendix results)
├── results/                           CSV outputs of the runs reported in the paper
│   ├── UAV1/
│   ├── UAV2_AccessPoint/
│   └── UAV3_GCS/
├── analysis/                          rebuild the paper's numbers and figures from results/
│   ├── recompute.py                   every number in the Results section
│   ├── make_tables.py                 data-heavy LaTeX tables and plot coordinates
│   ├── make_figures.py                tightening-ratio figure
│   └── make_heatmaps.py               class-by-generator heatmap and budget-sweep figure
├── data/README.md                     how to obtain the UAV-NIDD dataset
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
pip install -r requirements.txt
```

**Reproduce the paper's numbers and figures from the saved runs** (no GPU, no dataset needed):

```bash
python analysis/recompute.py      # prints the Results-section numbers
python analysis/make_tables.py    # writes analysis/fragments.txt
python analysis/make_figures.py   # writes figures/res_tightening.pdf
python analysis/make_heatmaps.py  # writes figures/res_class_gen_heatmap.pdf, figures/res_gen_sweep.pdf
```

**Re-run the full pipeline** (CNN-1D detector, oracle, seven constrained generators,
two decision-based baselines, Tri-Gate filtering, Phase-4 ranges):

1. Download UAV-NIDD (see [`data/README.md`](data/README.md)).
2. Open the notebook for the topology and set the dataset path in the CONFIG cell
   (`GLOB` in the UAV1 notebook; the `glob` entry of `DATASETS` in the UAV2/UAV3 notebooks).
   The notebooks were run on Kaggle, so the default paths start with `/kaggle/input/`.
3. Run all cells. Outputs are written to `cbag_out*/`, with the same file names as in `results/`.

The heavy generators (C-BO, CG-Diffusion, CC-CGAN, HopSkipJump, Sign-OPT) are enabled
by `ENABLE_HEAVY=True`; C-BO, HopSkipJump and Sign-OPT are capped at 60 seeds per class.
A GPU is recommended for the full run.

## Result files

Each topology folder in `results/` contains:

| File | Content |
|---|---|
| `dataset_summary.csv` | Train/test rows per class after preprocessing |
| `target_metrics.csv`, `target_perclass_metrics.csv` | CNN-1D detector performance on clean traffic (UAV1) |
| `oracle_metrics.csv` | Functionality-oracle accuracy and macro-F1 |
| `DIFS_scores.csv` | Dual-impact feature scores (MPI, FPC) used for the detector diagnostic (UAV1) |
| `cmr_eps.csv` | Cluster-membership radius per class |
| `perclass_gate.csv` | Gate attrition per class and generator (candidates, evasive, functional, FVAS) |
| `T7_constrained_vs_baseline.csv` | Per-generator totals, FVR and FVAS per 1,000 queries |
| `gate3_L1_sweep.csv` | FVAS per class and generator across the Gate-3 budget sweep |
| `T3_reachability.csv` | Operating and strict (CMR) survivor totals per budget |
| `T8_perturbation_dist.csv` | L1/L0 perturbation distribution per generator |
| `T9_feature_perturbation.csv` | Per-feature displacement in the survivor sets |
| `A6_ranges.csv` | Rulebook interval vs. survivor range per class and feature (Phase 4) |
| `T5_tightening.csv` | Median tightening ratio per class |
| `survivors.csv` | Survivor set at the operating budget (feature values, detector and oracle scores) |
| `survivors_budget_*.csv` | Survivor sets at the other budgets of the sweep |
| `run_config.csv` | Run settings (UAV2, UAV3) |

Operating budgets: τ = 0.25 (UAV1), 0.1 (UAV2), 10 (UAV3).

## Responsible use

The detectors attacked here are research models trained on a public dataset. No
deployed or third-party system was probed. Reproduce this work on isolated testbeds only.

## Citation

The citation will be added when the paper is published. The dataset should be cited as
described in [`data/README.md`](data/README.md).

## License

Code: MIT (see `LICENSE`). The UAV-NIDD dataset is distributed by its authors under CC BY 4.0.
