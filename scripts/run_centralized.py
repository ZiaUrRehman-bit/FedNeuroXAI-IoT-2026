"""
Centralised upper bound (Table 5): the same pipeline with a single client
holding the whole pool. With C=1, FedAvg and sufficient-statistic
aggregation reduce exactly to a centralised fit, so no separate training
code is involved.

    python scripts/run_centralized.py

Output: <out>/centralized.json
"""
import json
import os

from _common import base_parser, config_bins_epochs
from fedneuroxai import config
from fedneuroxai.client import explain_client
from fedneuroxai.experiment import build_federation, set_seed


def main():
    args = base_parser(__doc__).parse_args()
    results = []
    for ds in args.datasets:
        n_bins, epochs = config_bins_epochs(ds)
        for m in args.models:
            set_seed(args.seed)
            _, clients, X_h, y_h, scaler, model = build_federation(
                ds, m, n_clients=1, seed=args.seed, partition="dirichlet")
            acc = model.score(scaler.transform(X_h.values), y_h.values)
            exp = explain_client(clients[0], model, scaler, epochs=epochs, n_bins=n_bins,
                                 min_support=config.MIN_SUPPORT,
                                 min_confidence=config.MIN_CONFIDENCE,
                                 min_zhang=config.MIN_ZHANG)
            fm = exp["fidelity_metrics"] or {}
            r = {
                "dataset": ds, "model": m, "n_train": len(clients[0]["X_train"]),
                "n_test": len(clients[0]["X_test"]), "n_rules": len(exp["rules"]),
                "fidelity": fm.get("fidelity"), "completeness": fm.get("completeness"),
                "acc_holdout": round(float(acc), 4),
            }
            results.append(r)
            print(f"[{ds}/{m}] rules={r['n_rules']} fidelity={r['fidelity']} acc={r['acc_holdout']}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "centralized.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
