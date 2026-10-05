"""
Installation check (about 1 minute on a laptop CPU). Runs one small
end-to-end experiment and confirms that the deterministic part of the
pipeline reproduces the paper exactly.

    python scripts/smoke_test.py
"""
import json
import os

from _common import REPO_ROOT
from fedneuroxai.experiment import run_one, set_seed


def main():
    set_seed(42)
    r = run_one("diabetes", "logistic_regression", partition="dirichlet", n_instances_to_explain=1)
    ref = json.load(open(os.path.join(REPO_ROOT, "results", "noniid", "diabetes",
                                      "logistic_regression.json"), encoding="utf-8"))

    checks = {
        "global accuracy": (r["global_metrics"]["accuracy"], ref["global_metrics"]["accuracy"]),
        "global F1": (r["global_metrics"]["f1"], ref["global_metrics"]["f1"]),
        "client split sizes": ([(c["n_train"], c["n_test"]) for c in r["per_client"]],
                               [(c["n_train"], c["n_test"]) for c in ref["per_client"]]),
    }
    ok = True
    for name, (got, want) in checks.items():
        same = got == want
        ok &= same
        print(f"[{'OK ' if same else 'FAIL'}] {name}: {got}" + ("" if same else f" (paper: {want})"))

    n_rules = [c["n_rules"] for c in r["per_client"]]
    print(f"[info] rules mined per client: {n_rules} (seeded, so repeatable on one machine; "
          f"differs from the paper's unseeded run: {[c['n_rules'] for c in ref['per_client']]})")
    print("\nSmoke test passed." if ok else "\nSmoke test FAILED -- check package versions (README).")


if __name__ == "__main__":
    main()
