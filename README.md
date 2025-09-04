# BH<sup>3</sup>-MedRec
BH<sup>3</sup>-MedRec: Bilateral Hierarchical Heterogeneous Hypergraph Convolution Network for Medication Recommendation

## Overview
This repository provides the official PyTorch implementation and reproduction for our model BH<sup>3</sup>-MedRec. To ensure a fair review process, we are releasing a self-contained subset of the source code. Certain core implementation files related to our ongoing research are temporarily not included and will be released upon acceptance or at the conclusion of the review. Clear instructions are provided below to replicate the main tables and figures. For inquiries, please contact the corresponding author.

## Installation

1. Clone this git repository and change directory to this repository:
```python
cd BH3-MedRec/
```

2. A new conda environment is suggested.
```python
conda create --name BH3-MedRec
```

3. Activate the newly created environment.
```python
conda activate BH3-MedRec
```

4. Requirements
- torch==2.0.0
- numpy==1.24.2
- dill==0.3.8
- scikit-learn==1.3.2

## Download the data

1. You must have obtained access to MIMIC-III and MIMIC-IV databases before running the code. You can get access through the official link: [https://mimic.mit.edu/](https://mimic.mit.edu/).
2. Download the MIMIC-III and MIMIC-IV datasets, then unzip and put them in the data/input/ directory. Specifically, you need to download the following files diagnoses_icd.csv, procedures_icd.csv, and prescriptions.csv.
3. Download the [drugbank_drugs_info.csv](https://drive.google.com/file/d/1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk/view?usp=sharing) and [drug-DDI.csv](https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view) files, and put them in the data/input/ directory.

## Data Preprocessing
1. The division of departments is based on the curr_service field from the [services](https://mimic.mit.edu/docs/iv/modules/hosp/services/) table.
2. The data used in this experiment are from MIMIC-III-CMED/CSURG (CCU), MIMIC-III-MED, MIMIC-IV-PSYCH, and MIMIC-IV-OMED.
3. The data preprocessing code for mimic-iii dataset is provided in data/process_mimic-iii.py.
4. The data preprocessing code for mimic-iv dataset is provided in data/process_mimic-iv.py.

### Run the Code
```python
python BH3-MedRec.py
```