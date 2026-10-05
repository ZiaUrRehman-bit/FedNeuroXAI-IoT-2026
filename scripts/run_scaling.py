"""
Client-count scaling (Table 10(c)): Heart Disease federated across
C = 5, 10, 20 clients, non-IID and IID, all 5 classifiers. Records accuracy,
fidelity, usable clients and per-client rule-extraction time. SHAP/LIME are
skipped. Times depend on your hardware.

    python scripts/run_scaling.py
    python scripts/run_scaling.py --clients 5 10 20 40

Output: <out>/scaling.json
"""
import json
import os
import time

import numpy as np

from _common import base_parser
from fedneuroxai import client as fl_client
from fedneuroxai import experiment
from fedneuroxai.experiment import run_one, set_seed


def main():
    p = base_parser(__doc__)
    p.set_defaults(datasets=["heart"])
    p.add_argument("--clients", nargs="+", type=int, default=[5, 10, 20])
    args = p.parse_args()

    # Time each client's rule extraction by wrapping the function run_one calls.
    times = []
    original = fl_client.explain_client

    def timed(*a, **kw):
        t0 = time.perf_counter()
        out = original(*a, **kw)
        times.append(time.perf_counter() - t0)
        return out

    experiment.explain_client = timed

    log = []
    for ds in args.datasets:
        for part in ("dirichlet", "iid"):
            for C in args.clients:
                for m in args.models:
                    times.clear()
                    set_seed(args.seed)
                    t0 = time.perf_counter()
                    entry = {"dataset": ds, "partition": part, "n_clients": C, "model": m}
                    try:
                        r = run_one(ds, m, partition=part, n_clients=C, seed=args.seed,
                                    n_instances_to_explain=0)
                        pcs = r["per_client"]
                        usable = [c for c in pcs if c["fidelity_metrics"] is not None]
                        entry.update(
                            status="ok",
                            accuracy=r["global_metrics"]["accuracy"],
                            usable=len(usable),
                            fidelity=float(np.mean([c["fidelity_metrics"]["fidelity"] for c in usable]))
                            if usable else None,
                            coverage=float(np.mean([c["fidelity_metrics"]["completeness"] for c in usable]))
                            if usable else None,
                            mean_rules=float(np.mean([c["n_rules"] for c in pcs])),
                            zero_rule_clients=sum(c["n_rules"] == 0 for c in pcs),
                            mean_n_train=float(np.mean([c["n_train"] for c in pcs])),
                            min_n_train=int(min(c["n_train"] for c in pcs)),
                            mean_client_time_s=float(np.mean(times)),
                            max_client_time_s=float(np.max(times)),
                        )
                    except Exception as e:
                        entry.update(status="error", error=str(e))
                        print(f"!!! FAILED {part} C={C} {m}: {e}")
                    entry["wall_s"] = round(time.perf_counter() - t0, 1)
                    log.append(entry)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "scaling.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone: {sum(e['status'] == 'ok' for e in log)}/{len(log)} runs succeeded -> {path}")


if __name__ == "__main__":
    main()
