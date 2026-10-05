"""
Main experiment matrix: 4 datasets x 5 classifiers, C=5 clients, non-IID
(Dirichlet alpha=0.5) and/or IID partitions. Federated training, per-client
rule mining and fidelity, and the per-instance rule/SHAP/LIME comparison.

Produces the data behind Tables 3, 4, 6 (LIME part) and 9 and Figure 4.

    python scripts/run_experiments.py                       # both partitions
    python scripts/run_experiments.py --partition dirichlet
    python scripts/run_experiments.py --datasets heart --models lda

Output: <out>/noniid/<dataset>/<model>.json and <out>/iid/<dataset>/<model>.json
"""
import json
import os
import time
import traceback

from _common import base_parser
from fedneuroxai.experiment import run_one, set_seed

TAG = {"dirichlet": "noniid", "iid": "iid"}


def main():
    p = base_parser(__doc__)
    p.add_argument("--partition", choices=["dirichlet", "iid", "both"], default="both")
    args = p.parse_args()
    partitions = ["dirichlet", "iid"] if args.partition == "both" else [args.partition]

    log = []
    for part in partitions:
        out_dir = os.path.join(args.out, TAG[part])
        for ds in args.datasets:
            for m in args.models:
                set_seed(args.seed)
                t0 = time.time()
                entry = {"partition": part, "dataset": ds, "model": m}
                try:
                    r = run_one(ds, m, partition=part, seed=args.seed, out_dir=out_dir)
                    entry.update(status="ok", global_metrics=r["global_metrics"])
                except Exception as e:  # one failed combination must not stop the sweep
                    entry.update(status="failed", error=str(e), traceback=traceback.format_exc())
                    print(f"!!! FAILED {part}/{ds}/{m}: {e}")
                entry["seconds"] = round(time.time() - t0, 1)
                log.append(entry)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_experiments_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    ok = sum(e["status"] == "ok" for e in log)
    print(f"\nDone: {ok}/{len(log)} runs succeeded. Results under {args.out}")


if __name__ == "__main__":
    main()
