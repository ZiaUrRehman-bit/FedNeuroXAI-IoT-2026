"""
One end-to-end experiment for a (dataset, classifier, partition):

  1. split the data across simulated clients
  2. federated training of the global model h_g
  3. per-client rule mining and fidelity (global explanation)
  4. per-instance rule vs SHAP vs LIME (local explanation)
"""
import json
import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from . import config
from .client import explain_client
from .data import load_dataset
from .instance_explain import explain_instance
from .partition import make_client_splits
from .server import federated_train, fit_federated_scaler


def set_seed(seed: int):
    """Seed Python, NumPy and PyTorch. Note: the paper's original runs did
    not seed PyTorch, so PyAerial's autoencoder initialisation (and hence
    rule mining) varied slightly between runs; see the README."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def build_federation(dataset_key, model_key, n_clients=config.N_CLIENTS,
                     alpha=config.DIRICHLET_ALPHA, n_rounds=config.N_ROUNDS,
                     seed=config.SEED, partition="dirichlet"):
    """Steps 1-2. Returns (X, clients, X_holdout, y_holdout, scaler, global_model)."""
    X, y, _ = load_dataset(dataset_key)
    clients, X_holdout, y_holdout = make_client_splits(
        X, y, n_clients=n_clients, alpha=alpha, seed=seed, partition=partition,
        held_out_test_size=config.HELD_OUT_TEST_SIZE,
        client_local_test_size=config.CLIENT_LOCAL_TEST_SIZE,
    )
    scaler = fit_federated_scaler(clients)
    client_Xs = [scaler.transform(c["X_train"].values) for c in clients]
    client_ys = [c["y_train"].values for c in clients]
    global_model = federated_train(model_key, client_Xs, client_ys, n_rounds=n_rounds, seed=seed)
    return X, clients, X_holdout, y_holdout, scaler, global_model


def run_one(dataset_key, model_key, partition="dirichlet", n_clients=config.N_CLIENTS,
            alpha=config.DIRICHLET_ALPHA, n_rounds=config.N_ROUNDS, seed=config.SEED,
            n_instances_to_explain=config.N_INSTANCES_TO_EXPLAIN,
            min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE,
            min_zhang=config.MIN_ZHANG, out_dir=None, client_hook=None):
    """Run one experiment and optionally save it as <out_dir>/<dataset>/<model>.json.

    client_hook(client_index, explanation) is called after each client's rule
    mining (used by the scaling script to time it).
    """
    ds_cfg = config.DATASETS[dataset_key]
    n_bins, epochs = ds_cfg["n_bins"], ds_cfg["epochs"]

    X, clients, X_holdout, y_holdout, scaler, global_model = build_federation(
        dataset_key, model_key, n_clients=n_clients, alpha=alpha, n_rounds=n_rounds,
        seed=seed, partition=partition,
    )
    y_pred = global_model.predict(scaler.transform(X_holdout.values))
    global_metrics = {
        "accuracy": round(float(accuracy_score(y_holdout, y_pred)), 4),
        "f1": round(float(f1_score(y_holdout, y_pred, average="weighted")), 4),
    }
    print(f"[{dataset_key}/{model_key}/{partition}] global model on hold-out: {global_metrics}")

    per_client = []
    for ci, client in enumerate(clients):
        explanation = explain_client(
            client, global_model, scaler, epochs=epochs, n_bins=n_bins,
            min_support=min_support, min_confidence=min_confidence, min_zhang=min_zhang,
        )
        if client_hook is not None:
            client_hook(ci, explanation)
        fid = explanation["fidelity_metrics"]
        n_rules = len(explanation["rules"])
        print(f"  client {ci}: {len(client['X_train'])} train / {len(client['X_test'])} test, "
              f"{n_rules} rules" + (f", fidelity={fid['fidelity']:.3f}" if fid else ", no local test split"))

        instances = []
        disc_test = explanation["disc_test"]
        if len(disc_test) and n_rules > 0 and n_instances_to_explain > 0:
            train_scaled = scaler.transform(client["X_train"].values)
            for idx in range(min(n_instances_to_explain, len(disc_test))):
                inst_scaled = scaler.transform(client["X_test"].iloc[[idx]].values)[0]
                instances.append(explain_instance(
                    global_model, model_key, explanation["rules"], disc_test.iloc[idx],
                    inst_scaled, train_scaled, feature_names=list(X.columns),
                    class_names=list(np.unique(global_model.classes_)),
                    top_k=config.TOP_K_OVERLAP,
                ))

        per_client.append({
            "client_id": ci,
            "n_train": len(client["X_train"]),
            "n_test": len(client["X_test"]),
            "n_rules": n_rules,
            "fidelity_metrics": fid,
            "instance_explanations": instances,
        })

    result = {
        "dataset": dataset_key, "model": model_key, "partition": partition,
        "n_clients": n_clients, "alpha": alpha, "n_rounds": n_rounds, "seed": seed,
        "n_bins": n_bins, "rule_epochs": epochs,
        "global_metrics": global_metrics,
        "per_client": per_client,
    }

    if out_dir:
        path = os.path.join(out_dir, dataset_key, f"{model_key}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  saved -> {path}")
    return result
