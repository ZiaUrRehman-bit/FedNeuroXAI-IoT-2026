# Datasets

The four clinical tabular datasets used in the paper, as CSV files exactly as
used for the reported results. All are binary classification tasks.

| File | Dataset | Rows | Features | Target column | Source |
|---|---|---|---|---|---|
| `diabetes.csv` | Pima Indians Diabetes | 768 | 8 | `target` (1 = diabetic) | Smith et al. (1988); UCI ML Repository / OpenML |
| `wisconsin_breast_cancer.csv` | Breast Cancer Wisconsin (Diagnostic) | 569 | 30 | `target` (0 = malignant, 1 = benign) | Wolberg et al., UCI ML Repository, doi:10.24432/C5DW2B |
| `heart11.csv` | Heart Disease Dataset (Comprehensive) | 1,190 | 11 | `target` (1 = disease) | M. Siddhartha, IEEE DataPort (2020), doi:10.21227/dz4t-cm36 |
| `FetalHeartRate.csv` | Fetal Heart Rate Features of Healthy and Late IUGR Fetuses | 262 | 31 | `Clinical_Diagnosis` | Pini et al., IEEE DataPort (2020), doi:10.21227/mzc6-jt52 |

Feature names are sanitised at load time (`fedneuroxai/data.py`), so a column
such as `chest pain type` appears as `chest_pain_type` in mined rules.

## Licences and attribution

The datasets are redistributed here only for reproducibility and remain
under their original terms; they are not covered by this repository's MIT
licence. Please cite the original sources above when you use them.

> **Before making this repository public:** confirm that each source permits
> redistribution. If one does not, delete that CSV from `data/` and add a
> note here telling users to download it from the DOI above and save it
> under the same file name. The code needs no other change.
