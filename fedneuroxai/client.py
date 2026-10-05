"""
Client-side code: everything that runs on a hospital / edge node using only
its own local data.

Federation layer (Section 3.2):
  - local feature statistics for the shared scaler
  - local model updates (one epoch of SGD per FedAvg round)
  - local sufficient statistics for Gaussian Naive Bayes and LDA

Explanation layer (Sections 3.3-3.4), run after the global model h_g has
been received:
  - equal-frequency discretisation of the local data
  - surrogate labelling with h_g's own predictions
  - PyAerial rule mining and local fidelity evaluation

Only model parameters or aggregate statistics ever leave the client; rules,
narratives and per-prediction explanations stay local.
"""
import numpy as np
import pandas as pd
from aerial import discretization
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from .rules import evaluate, extract_and_filter, keep_class_rules


# ---------------------------------------------------------------------------
# Federation layer
# ---------------------------------------------------------------------------

def local_feature_stats(X_train: pd.DataFrame):
    """(n, per-feature mean, per-feature variance) sent to the server so it
    can build one shared StandardScaler without seeing any rows."""
    values = X_train.values
    return float(len(values)), values.mean(axis=0), values.var(axis=0)


def build_local_model(model_key: str, seed: int):
    """A client's local copy of a gradient-trained model."""
    if model_key == "logistic_regression":
        return SGDClassifier(loss="log_loss", penalty="l2", random_state=seed)
    if model_key == "linear_svm":
        # modified_huber (not hinge) so the SVM exposes predict_proba,
        # which LIME needs.
        return SGDClassifier(loss="modified_huber", penalty="l2", random_state=seed)
    if model_key == "mlp":
        # warm_start is deliberately not set: with partial_fit it requires
        # every call to see the same classes, which breaks on single-class
        # non-IID client shards.
        return MLPClassifier(hidden_layer_sizes=(32,), max_iter=1, random_state=seed)
    raise ValueError(f"{model_key} is trained via sufficient statistics, not FedAvg")


def local_update(model, X, y, classes, first_round: bool):
    """One local epoch of SGD on the client's own data (one FedAvg round)."""
    if first_round:
        model.partial_fit(X, y, classes=classes)
    else:
        model.partial_fit(X, y)
    return model


def local_gaussian_nb(X, y):
    """Local Gaussian NB fit; the server reads only its per-class
    (count, mean, variance) from class_count_, theta_ and var_."""
    m = GaussianNB()
    m.fit(X, y)
    return m


def local_lda_stats(X, y):
    """Local LDA fit plus the per-class (count, mean, scatter matrix) the
    server needs; sklearn's LDA does not expose these directly."""
    m = LinearDiscriminantAnalysis(store_covariance=True)
    m.fit(X, y)
    m._class_n, m._class_mean, m._class_scatter = [], [], []
    for c in m.classes_:
        Xc = X[y == c]
        mean_c = Xc.mean(axis=0)
        diff = Xc - mean_c
        m._class_n.append(float(len(Xc)))
        m._class_mean.append(mean_c)
        m._class_scatter.append(diff.T @ diff)
    return m


# ---------------------------------------------------------------------------
# Explanation layer
# ---------------------------------------------------------------------------

def discretize_client(X_train: pd.DataFrame, X_test: pd.DataFrame, n_bins: int = 4):
    """Equal-frequency binning fitted on this client's own data only."""
    combined = pd.concat([X_train, X_test], axis=0)
    disc_all = discretization.equal_frequency_discretization(combined.copy(), n_bins=n_bins)
    for col in disc_all.columns:
        disc_all[col] = disc_all[col].astype("category")
    disc_train = disc_all.iloc[:len(X_train)].reset_index(drop=True)
    disc_test = disc_all.iloc[len(X_train):].reset_index(drop=True)
    return disc_train, disc_test


def explain_client(client: dict, global_model, scaler, epochs: int = 10, n_bins: int = 4,
                   min_support: float = 0.10, min_confidence: float = 0.80,
                   min_zhang: float = 0.30) -> dict:
    """Mine and evaluate this client's IF-THEN rule set for the global model
    (Algorithm 1, lines 1-8).

    client: {"X_train", "y_train", "X_test", "y_test"} raw local splits.
    scaler: the shared federated scaler the global model was trained with.
    """
    X_train_raw, X_test_raw = client["X_train"], client["X_test"]

    # Surrogate labels: the global model's own predictions on local data.
    y_surrogate_train = global_model.predict(scaler.transform(X_train_raw.values))
    y_surrogate_test = np.array([])
    if len(X_test_raw):
        y_surrogate_test = global_model.predict(scaler.transform(X_test_raw.values))

    disc_train, disc_test = discretize_client(X_train_raw, X_test_raw, n_bins=n_bins)
    disc_train["class"] = y_surrogate_train
    if len(disc_test):
        disc_test["class"] = y_surrogate_test

    df_rules, stats = extract_and_filter(
        disc_train, epochs=epochs, min_support=min_support,
        min_confidence=min_confidence, min_zhang=min_zhang,
    )
    df_rules = keep_class_rules(df_rules)

    fidelity_metrics = None
    if len(disc_test):
        fidelity_metrics = evaluate(
            df_rules, disc_test, y_surrogate_test,
            all_classes=tuple(np.unique(global_model.classes_)),
        )

    return {
        "rules": df_rules,
        "aerial_stats": stats,
        "fidelity_metrics": fidelity_metrics,
        "disc_train": disc_train,
        "disc_test": disc_test,
    }
