"""
Seed robustness (Table 7): reruns the non-IID matrix under several seeds.
Each seed redraws the Dirichlet partition and the model initialisation.
SHAP/LIME are skipped (only fidelity and accuracy are needed).

    python scripts/run_seeds.py                    # seeds 42-46
    python scripts/run_seeds.py --seeds 42 43

Output: <out>/seeds/seed<N>/<dataset>/<model>.json
Some (dataset, model, seed) combinations fail on degenerate shards (in the
paper: Fetal HR at seed 46, Heart LDA at two seeds); failures are logged.
"""
import json
import os
import traceback

from _common import base_parser
from fedneuroxai.experiment import run_one, set_seed


def main():
    p = base_parser(__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = p.parse_args()

    log = []
    for seed in args.seeds:
        out_dir = os.path.join(args.out, "seeds", f"seed{seed}")
        for ds in args.datasets:
            for m in args.models:
                set_seed(seed)
                try:
                    run_one(ds, m, partition="dirichlet", seed=seed, out_dir=out_dir,
                            n_instances_to_explain=0)
                    log.append({"seed": seed, "dataset": ds, "model": m, "status": "ok"})
                except Exception as e:
                    print(f"!!! FAILED seed={seed} {ds}/{m}: {e}")
                    log.append({"seed": seed, "dataset": ds, "model": m, "status": "failed",
                                "error": str(e), "traceback": traceback.format_exc()})

    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    with open(os.path.join(args.out, "seeds", "run_seeds_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone: {sum(e['status'] == 'ok' for e in log)}/{len(log)} runs succeeded.")


if __name__ == "__main__":
    main()
