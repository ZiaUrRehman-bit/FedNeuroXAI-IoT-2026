"""
Central configuration: paths, datasets, classifiers and the experimental
settings reported in the paper. Every script reads its defaults from here,
so changing a value in this file changes it everywhere.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

# Local LLM used for the narrative layer (Section 3.5 of the paper).
# Either a Hugging Face model id or a local directory; override with the
# FEDNEUROXAI_LLM environment variable or the --llm argument of the scripts.
LLM_PATH = os.environ.get("FEDNEUROXAI_LLM", "meta-llama/Llama-3.2-3B-Instruct")

# dataset key -> CSV file, target column, display label, (bins B, epochs e)
DATASETS = {
    "diabetes": {
        "file": "diabetes.csv", "target": "target", "label": "Diabetes",
        "n_bins": 4, "epochs": 15,
    },
    "wbc": {
        "file": "wisconsin_breast_cancer.csv", "target": "target", "label": "WBC",
        "n_bins": 3, "epochs": 15,
    },
    "heart": {
        "file": "heart11.csv", "target": "target", "label": "Heart",
        "n_bins": 4, "epochs": 10,
    },
    "fetalheartrate": {
        "file": "FetalHeartRate.csv", "target": "Clinical_Diagnosis", "label": "Fetal HR",
        "n_bins": 4, "epochs": 20,
    },
}

MODEL_KEYS = ["logistic_regression", "linear_svm", "mlp", "gaussian_nb", "lda"]
MODEL_LABELS = {
    "logistic_regression": "LR", "linear_svm": "SVM", "mlp": "MLP",
    "gaussian_nb": "NB", "lda": "LDA",
}

# Paper settings (Section 4).
N_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
N_ROUNDS = 100                 # FedAvg rounds (no-op for NB / LDA)
SEED = 42
HELD_OUT_TEST_SIZE = 0.3       # central hold-out for global accuracy
CLIENT_LOCAL_TEST_SIZE = 0.3   # per-client split for rule fidelity
MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.60
MIN_ZHANG = 0.05
N_INSTANCES_TO_EXPLAIN = 3     # per client, for the SHAP/LIME comparison
TOP_K_OVERLAP = 3              # k in the Jaccard overlap (Eq. 7)
NARRATIVE_TOP_RULES = 6        # kappa: top rules per class sent to the LLM
