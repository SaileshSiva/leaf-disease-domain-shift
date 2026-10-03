# leaf-disease-domain-shift

Master's in DS and AI Capstone Project (CS5998), University of Moratuwa — Plant leaf disease classification.

## What this project asks

Leaf disease classifiers trained on PlantVillage report accuracies above 99%, but Mohanty, Hughes and Salathé (2016) found the same model scored 31.4% on images photographed under conditions unlike the training data. PlantVillage shows one detached leaf on a plain background; a field photograph has soil, overlapping leaves, shadows and blur.

This project measures that lab-to-field gap on a fixed set of tomato and potato disease classes, and tests which training decisions close the most of it.

## Status

| | |
|---|---|
| Submission 01 — Project Definition | Submitted ([document](reports/Submission_01_Project_Definition.pdf)) |
| Submission 02 — Technical Checkpoint | In progress |
| Submission 03 — Final Submission | Not started |

## Data

| Dataset | Role | Source |
|---|---|---|
| PlantVillage | Training, validation, in-domain test | [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) |
| PlantDoc | Out-of-domain (field) test | [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) |

Neither dataset is committed to this repository.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Results

Not yet available. Populated as experiments are run.
