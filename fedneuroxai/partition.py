"""
Simulated federation: splitting a dataset across clients (Section 3.1).

  1. A central held-out set, never seen by any client, used only to report
     the global model's accuracy.
  2. The remaining pool split across C clients, either non-IID (per-class
     Dirichlet, Eq. 1) or IID (stratified, equal class proportions).
  3. Each client's shard split into a local training part (federated
     training + rule mining) and a local test part (rule fidelity).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def dirichlet_partition(y, n_clients=5, alpha=0.5, seed=42):
    """Per-class Dirichlet label-skew split (Hsu et al., 2019)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    client_indices = [[] for _ in range(n_clients)]

    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx_c, split_points)):
            client_indices[k].extend(part.tolist())

    return [np.array(sorted(idxs)) for idxs in client_indices]


def iid_partition(y, n_clients=5, seed=42):
    """Stratified split: every client gets the pool's class proportions."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    return [test_idx for _, test_idx in skf.split(np.zeros(len(y)), y)]


def make_client_splits(X, y, n_clients=5, alpha=0.5, held_out_test_size=0.3,
                       client_local_test_size=0.3, seed=42, partition="dirichlet"):
    """Returns (clients, X_holdout, y_holdout).

    clients: list of dicts with keys X_train, y_train, X_test, y_test.
    A client whose shard is too small or single-class to split (stratified
    splitting needs >= 2 members per class) keeps all its rows for training
    and gets an empty local test split; its fidelity is then undefined and
    it is excluded from fidelity means ("usable" clients in the paper).
    """
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=held_out_test_size, random_state=seed, stratify=y,
    )
    X_pool = X_pool.reset_index(drop=True)
    y_pool = y_pool.reset_index(drop=True)

    if partition == "iid":
        client_idx = iid_partition(y_pool.values, n_clients=n_clients, seed=seed)
    elif partition == "dirichlet":
        client_idx = dirichlet_partition(y_pool.values, n_clients=n_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError(f"unknown partition mode: {partition!r}")

    clients = []
    for idx in client_idx:
        Xk = X_pool.iloc[idx].reset_index(drop=True)
        yk = y_pool.iloc[idx].reset_index(drop=True)

        min_class_count = pd.Series(yk).value_counts().min() if len(yk) else 0
        if len(np.unique(yk)) < 2 or len(yk) < 10 or min_class_count < 2:
            clients.append({
                "X_train": Xk, "y_train": yk,
                "X_test": Xk.iloc[0:0], "y_test": yk.iloc[0:0],
            })
            continue

        Xk_tr, Xk_te, yk_tr, yk_te = train_test_split(
            Xk, yk, test_size=client_local_test_size, random_state=seed, stratify=yk,
        )
        clients.append({
            "X_train": Xk_tr.reset_index(drop=True), "y_train": yk_tr.reset_index(drop=True),
            "X_test": Xk_te.reset_index(drop=True), "y_test": yk_te.reset_index(drop=True),
        })

    return clients, X_holdout.reset_index(drop=True), y_holdout.reset_index(drop=True)
