# Dataset Documentation & Provenance

## Dataset Name
**Diabetes 130-US Hospitals for Years 1999–2008 Data Set**

## Citation & Academic Provenance
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, K. J. Cios, and John N. Clore.  
*"Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records."*  
BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014. DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670).

## Source URL & Licensing
- **Repository:** UCI Machine Learning Repository
- **URL:** [https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Direct Download:** `https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip`
- **License / Terms of Use:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Privacy & De-identification:** The dataset is fully anonymized and de-identified in strict compliance with HIPAA Safe Harbor standards. No protected health information (PHI) is present.

## Dataset Dimensions & Attributes
- **Raw Encounter Records:** 101,766 inpatient encounters across 130 US hospitals.
- **Unique Patients:** 71,518 unique patients.
- **Attributes:** 50 clinical, demographic, utilization, diagnostic, and administrative variables.
- **Target Variable (`readmitted`):**
  - `<30`: Patient readmitted within 30 days of discharge (Positive Class, ~11.2% in raw data, ~8.98% in cleansed primary encounters).
  - `>30`: Patient readmitted after 30 days of discharge (Negative Class for 30-day quality benchmarking).
  - `NO`: No readmission recorded.

## Automated Download
To automatically download and unpack the dataset:
```bash
python src/data/download_dataset.py
```
This extracts `diabetic_data.csv` and `IDS_mapping.csv` into `data/raw/`.
