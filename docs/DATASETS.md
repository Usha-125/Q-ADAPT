# Datasets

The raw datasets are **not** redistributed. Download them from the official sources
listed below and place the CSV files under `data/raw/` (git-ignored).

| Dataset | Role in Q-ADAPT | Source |
|---|---|---|
| CSE-CIC-IDS2018 | primary ML training and evaluation | Canadian Institute for Cybersecurity / UNB: <https://www.unb.ca/cic/datasets/ids-2018.html> (also on the AWS Open Data Registry) |
| CIC-IDS2017 | cross-dataset external validation | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| UNSW-NB15 | third validation set (different feature space) | UNSW Canberra: <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |
| NVD / CVE | vulnerability severity and exploitability | NVD CVE API 2.0: <https://services.nvd.nist.gov/rest/json/cves/2.0> |

## Layout

```
data/raw/cse-cic-ids2018/*.csv     # Processed Traffic Data for ML Algorithms
data/raw/cic-ids2017/*.csv         # MachineLearningCVE
data/raw/unsw-nb15/UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
```

## Usage

```bash
# train and save the detector on 2018 data (50k rows sampled per file)
qadapt train --csv data/raw/cse-cic-ids2018/*.csv --model random_forest

# Experiment 1, hold-out on 2018
python -m experiments.experiment1_ml --train-csv data/raw/cse-cic-ids2018/*.csv

# Experiment 1, train on 2018 and test on 2017 (cross-dataset)
python -m experiments.experiment1_ml \
    --train-csv data/raw/cse-cic-ids2018/*.csv --test-csv data/raw/cic-ids2017/*.csv
```

`ml_engine.datasets.canonicalize_cic` maps the two CIC spellings onto one canonical
feature schema, for example `Tot Fwd Pkts` (2018) and `Total Fwd Packets` (2017). It
also maps the labels onto the unified taxonomy and removes the repeated header rows found
in some 2018 files. UNSW-NB15 is loaded with its own numeric feature set
(`load_unsw_nb15`), because its features differ from CICFlowMeter's.

## What the datasets do *not* contain

IDS datasets contain no network topology, asset criticality, defense costs, patch times
or business-disruption data. Q-ADAPT does not pretend otherwise:

* **Layer 1 (ML)** uses the real datasets.
* **Layer 2 (decision)** uses controlled synthetic enterprises (`attack_graph/topology.py`)
  seeded with documented CVEs (`attack_graph/vulnerabilities.py`). The topologies come
  in small, medium and large sizes plus the fixed demo network.

## Synthetic flows

`ml_engine.synthetic.generate_flows` produces class-conditional, overlapping
CICFlowMeter-like flows. It is used for tests, CI, the dashboard's live-detection demo
and offline runs. **Metrics computed on synthetic flows validate the pipeline only. They
must not be reported as dataset results.** Experiment 1 labels its data source
explicitly in its output.
