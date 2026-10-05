"""
Table 6: for one representative client per dataset, the top-5 SHAP-ranked
features, their SHAP and LIME importance (each normalised to that client's
maximum) and the share of the client's own rules that use each feature.

SHAP is recomputed (mean |SHAP| over the client's local test rows). LIME
importance is read from the saved per-instance results (results/noniid or
results/iid), i.e. the same 3 explained instances behind Table 4. On Heart
Disease LIME returned no explanation for those instances, shown as "--".

    python scripts/make_shap_lime_table.py
    python scripts/make_shap_lime_table.py --results results_rerun

Output: <out>/shap_lime_table.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import shap

from _common import RERUN_DIR, config_bins_epochs
from fedneuroxai import config
from fedneuroxai.client import explain_client
from fedneuroxai.experiment import build_federation, set_seed
from fedneuroxai.instance_explain import LINEAR_MODEL_KEYS

# (dataset, client id, model, partition) -- the representative clients of Table 6
CONFIGS = [
    ("diabetes", 1, "logistic_regression", "dirichlet"),
    ("wbc", 0, "gaussian_nb", "iid"),
    ("heart", 1, "logistic_regression", "dirichlet"),
    ("fetalheartrate", 0, "logistic_regression", "dirichlet"),
]
TOP_K = 5


def mean_abs_shap(model, model_key, X_train_scaled, X_eval_scaled, feature_names):
    rng = np.random.RandomState(42)
    bg = X_train_scaled[rng.choice(len(X_train_scaled), size=min(50, len(X_train_scaled)), replace=False)]
    if model_key in LINEAR_MODEL_KEYS:
        sv = shap.LinearExplainer(model, bg).shap_values(X_eval_scaled)
    else:
        sv = shap.Explainer(model.predict_proba, bg)(X_eval_scaled).values
    sv = np.asarray(sv)
    if sv.ndim == 2:
        v = np.abs(sv).mean(axis=0)
    elif sv.shape[1] == len(feature_names):
        v = np.abs(sv).mean(axis=(0, 2))
    else:
        v = np.abs(sv).mean(axis=(0, 1))
    return pd.Series(v, index=feature_names)


def rule_usage(df_rules, feature_names):
    counts = pd.Series(0, index=feature_names)
    for ant in df_rules["antecedents"]:
        for feat in {cond.split("=")[0] for cond in str(ant).split(" AND ")}:
            if feat in counts.index:
                counts[feat] += 1
    return counts / max(len(df_rules), 1)


def saved_lime(results_root, ds, m, partition, cid, feature_names):
    tag = "noniid" if partition == "dirichlet" else "iid"
    d = json.load(open(os.path.join(results_root, tag, ds, f"{m}.json"), encoding="utf-8"))
    inst = d["per_client"][cid]["instance_explanations"]
    w = pd.Series(0.0, index=feature_names)
    ok = [i for i in inst if not i.get("lime_error") and i.get("lime_ranking")]
    if not ok:
        return None
    for i in ok:
        for feat, val in i["lime_ranking"].items():
            if feat in w.index:
                w[feat] += abs(val)
    return w / len(inst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=config.RESULTS_DIR, help="where the saved LIME results are")
    ap.add_argument("--out", default=RERUN_DIR)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    rows = []
    for ds, cid, m, part in CONFIGS:
        set_seed(args.seed)
        X, clients, _, _, scaler, model = build_federation(ds, m, seed=args.seed, partition=part)
        n_bins, epochs = config_bins_epochs(ds)
        c = clients[cid]
        exp = explain_client(c, model, scaler, epochs=epochs, n_bins=n_bins,
                             min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE,
                             min_zhang=config.MIN_ZHANG)
        feats = list(X.columns)
        eval_raw = c["X_test"] if len(c["X_test"]) else c["X_train"]
        shap_imp = mean_abs_shap(model, m, scaler.transform(c["X_train"].values),
                                 scaler.transform(eval_raw.values), feats)
        usage = rule_usage(exp["rules"], feats)
        lime = saved_lime(args.results, ds, m, part, cid, feats)
        top = shap_imp.sort_values(ascending=False).head(TOP_K).index
        lime_max = lime[top].max() if lime is not None and lime[top].max() > 0 else None
        for f in top:
            rows.append({
                "dataset": config.DATASETS[ds]["label"], "client": cid, "model": m,
                "n_rules": len(exp["rules"]), "feature": f,
                "shap_norm": round(float(shap_imp[f] / shap_imp.max()), 3),
                "lime_norm": None if lime_max is None else round(float(lime[f] / lime_max), 3),
                "rule_pct": round(float(usage[f]) * 100, 1),
            })

    print(f"\n{'Dataset':<10}{'Feature':<26}{'SHAP':>7}{'LIME':>7}{'Rules':>8}")
    for r in rows:
        lime = "--" if r["lime_norm"] is None else f"{r['lime_norm']:.3f}"
        print(f"{r['dataset']:<10}{r['feature']:<26}{r['shap_norm']:>7.3f}{lime:>7}{r['rule_pct']:>7.1f}%")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "shap_lime_table.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
