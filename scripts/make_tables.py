"""
Rebuild the paper's result tables from the JSON files in results/.

Works on the shipped results out of the box, or on your own reruns:
    python scripts/make_tables.py                      # shipped results
    python scripts/make_tables.py --results my_results # your reruns

Prints every table and writes them to <results>/tables.md.
Tables that need a missing input (e.g. no narrative run) are skipped.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedneuroxai import config  # noqa: E402

DS = list(config.DATASETS)
LABEL = {k: v["label"] for k, v in config.DATASETS.items()}
MODELS = config.MODEL_KEYS
ML = config.MODEL_LABELS


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def usable(run):
    """Clients with a local test split (fidelity defined)."""
    return [c for c in run["per_client"] if c["fidelity_metrics"] is not None]


def run_summary(run):
    u = usable(run)
    fm = [c["fidelity_metrics"] for c in u]
    inst = [i for c in run["per_client"] for i in c["instance_explanations"]]
    return {
        "acc": run["global_metrics"]["accuracy"],
        "f1": run["global_metrics"]["f1"],
        "usable": len(u),
        "fid": mean([m["fidelity"] for m in fm]),
        "cov": mean([m["completeness"] for m in fm]),
        "fid_cov": mean([m["fidelity"] / m["completeness"] for m in fm if m["completeness"] > 0]),
        # rule counts are averaged over ALL clients (a client without a local
        # test split still mines rules), as in the paper's #R / #Rules columns
        "n_rules": mean([c["n_rules"] for c in run["per_client"]]),
        "conf": mean([m["avg_confidence"] for m in fm]),
        "supp": mean([m["avg_support"] for m in fm]),
        "ovl": mean([m["overlap"] for m in fm]),
        "zh": mean([m["avg_zhang"] for m in fm]),
        "s_ovl": mean([i["rule_shap_overlap"] for i in inst]),
        "l_ovl": mean([i["rule_lime_overlap"] for i in inst]),
        "lime_fail": sum(1 for i in inst if i.get("lime_error")),
        "shap_fail": sum(1 for i in inst if i.get("shap_error")),
        "n_inst": len(inst),
        "fid_by_client": {c["client_id"]: c["fidelity_metrics"]["fidelity"] for c in u},
        "rules_by_client": {c["client_id"]: c["n_rules"] for c in u},
    }


def collect(root):
    out = {}
    for ds in DS:
        for m in MODELS:
            run = load(os.path.join(root, ds, f"{m}.json"))
            out[(ds, m)] = run_summary(run) if run else None
    return out


class Report:
    def __init__(self):
        self.lines = []

    def table(self, title, header, rows):
        self.lines += [f"### {title}", "", "| " + " | ".join(header) + " |",
                       "|" + "---|" * len(header)]
        self.lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        self.lines.append("")
        print(f"\n{title}")
        widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        for r in [header] + rows:
            print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=config.RESULTS_DIR)
    args = ap.parse_args()
    R = args.results
    rep = Report()

    noniid = collect(os.path.join(R, "noniid"))
    iid = collect(os.path.join(R, "iid"))

    # Table 3 -- accuracy and fidelity (non-IID)
    rows = []
    for ds in DS:
        for m in MODELS:
            s = noniid[(ds, m)]
            if s:
                rows.append([LABEL[ds], ML[m], fmt(s["acc"]), fmt(s["f1"]), fmt(s["fid"]),
                             fmt(s["cov"]), fmt(s["fid_cov"])])
    rep.table("Table 3: accuracy and rule fidelity (non-IID, mean over usable clients)",
              ["Dataset", "Clf.", "Acc", "F1", "Fid.", "Cov.", "Fid|Cov"], rows)

    # Table 4 -- rule-set quality (non-IID)
    rows = []
    for ds in DS:
        for m in MODELS:
            s = noniid[(ds, m)]
            if s:
                rows.append([LABEL[ds], ML[m], fmt(s["n_rules"], 0), fmt(s["conf"]), fmt(s["supp"]),
                             fmt(s["ovl"]), fmt(s["zh"]), f"{fmt(s['s_ovl'])}/{fmt(s['l_ovl'])}"])
    rep.table("Table 4: rule-set quality (non-IID; #R over all clients, metrics over usable "
              "clients, S/L Ovl over explained instances; paper rounds S/L Ovl to 2 d.p.)",
              ["Dataset", "Clf.", "#R", "Conf", "Supp", "Ovl", "Zh", "S/L Ovl"], rows)

    # SHAP/LIME failure counts (Section 5.3)
    rows = []
    for ds in DS:
        n = sum(noniid[(ds, m)]["n_inst"] for m in MODELS if noniid[(ds, m)])
        sf = sum(noniid[(ds, m)]["shap_fail"] for m in MODELS if noniid[(ds, m)])
        lf = sum(noniid[(ds, m)]["lime_fail"] for m in MODELS if noniid[(ds, m)])
        rows.append([LABEL[ds], n, sf, lf])
    rep.table("SHAP/LIME failures over explained instances (non-IID)",
              ["Dataset", "Instances", "SHAP failed", "LIME failed"], rows)

    # Table 9 -- IID vs non-IID
    def ds_means(coll, ds):
        ss = [coll[(ds, m)] for m in MODELS if coll[(ds, m)]]
        return {
            "usable": mean([s["usable"] for s in ss]),
            "fid": mean([s["fid"] for s in ss]),
            "rules": mean([s["n_rules"] for s in ss]),
            "acc": mean([s["acc"] for s in ss]),
        }

    rows = []
    t9 = {}
    for ds in DS:
        a, b = ds_means(noniid, ds), ds_means(iid, ds)
        t9[ds] = a
        rows.append([LABEL[ds], "Non-IID", fmt(a["usable"], 2), fmt(a["fid"]), fmt(a["rules"], 1), fmt(a["acc"])])
        rows.append(["", "IID", fmt(b["usable"], 2), fmt(b["fid"]), fmt(b["rules"], 1), fmt(b["acc"])])
        # IID restricted to the client ids usable under non-IID
        fids, rls = [], []
        for m in MODELS:
            n, i = noniid[(ds, m)], iid[(ds, m)]
            if n and i:
                keep = [cid for cid in n["fid_by_client"] if cid in i["fid_by_client"]]
                fids.append(mean([i["fid_by_client"][c] for c in keep]))
                rls.append(mean([i["rules_by_client"][c] for c in keep]))
        if b["usable"] and a["usable"] and b["usable"] > a["usable"]:
            rows.append(["", "Matched IID", fmt(a["usable"], 2), fmt(mean(fids)), fmt(mean(rls), 1), fmt(b["acc"])])
    rep.table("Table 9: IID vs non-IID (mean over 5 classifiers)",
              ["Dataset", "Partition", "Usable/5", "Fidelity", "#Rules", "Acc"], rows)

    # Table 5 -- centralised upper bound vs federated
    cen = load(os.path.join(R, "centralized.json"))
    if cen:
        rows = []
        for ds in DS:
            cs = [c for c in cen if c["dataset"].lower() == ds]
            rows.append([LABEL[ds], fmt(mean([c["fidelity"] for c in cs])), fmt(t9[ds]["fid"]),
                         fmt(mean([c["acc_holdout"] for c in cs])), fmt(t9[ds]["acc"])])
        rep.table("Table 5: centralised upper bound vs federated mean (non-IID)",
                  ["Dataset", "Fid.C", "Fid.F", "Acc.C", "Acc.F"], rows)

    # Table 7 -- seed robustness
    seed_dirs = sorted(glob.glob(os.path.join(R, "seeds", "seed*")))
    if seed_dirs:
        rows = []
        for ds in DS:
            fids, accs = [], []
            for sd in seed_dirs:
                coll = collect(sd)
                ss = [coll[(ds, m)] for m in MODELS if coll[(ds, m)]]
                if ss:
                    fids.append(mean([s["fid"] for s in ss]))
                    accs.append(mean([s["acc"] for s in ss]))
            rows.append([LABEL[ds], len(fids), f"{np.mean(fids):.3f} ± {np.std(fids):.3f}",
                         f"{np.mean(accs):.3f} ± {np.std(accs):.3f}"])
        rep.table("Table 7: mean ± std (population) across Dirichlet seeds (mean over classifiers per seed)",
                  ["Dataset", "Seeds", "Fidelity", "Acc"], rows)

    # Table 8 -- narrative faithfulness
    nar = load(os.path.join(R, "narrative_faithfulness.json"))
    if nar:
        rows = []
        name_map = {"Diabetes": "diabetes", "WBC": "wbc", "Heart": "heart", "FetalHeartRate": "fetalheartrate"}
        for ds in DS:
            rs = [r for r in nar["per_client"] if name_map.get(r["dataset"], r["dataset"]) == ds
                  and not r.get("skipped")]
            with_thr = [r for r in rs if r["n_thresholds_in_narrative"]]
            thr = sum(r["n_thresholds_in_narrative"] for r in rs)
            mat = sum(r["n_matched_verbatim"] for r in rs)
            rows.append([LABEL[ds], f"{len(with_thr)}/5", thr, mat, fmt(mat / thr if thr else None)])
        s = nar["summary"]
        rows.append(["All", f"{s['n_narratives_generated']}/20", s["total_thresholds_checked"],
                     s["total_matched_verbatim"], fmt(s["overall_faithfulness_rate"])])
        rep.table("Table 8: LLM narrative threshold faithfulness",
                  ["Dataset", "Narr.", "Thr.", "Matched", "Rate"], rows)

    # Table 10(b) -- resource footprint
    fp = load(os.path.join(R, "footprint.json"))
    if fp:
        rows = [["Rule extraction", f"{fp['rule_extraction']['wall_seconds']} s",
                 f"{fp['rule_extraction']['rss_delta_mb']} MB", f"{fp['disk_footprint']['mined_ruleset_kb']} KB"]]
        if "llm_load" in fp:  # absent when profiled with --no-llm
            rows += [
                ["LLM load", f"{fp['llm_load']['wall_seconds']} s", f"{fp['llm_load']['rss_delta_mb']} MB",
                 f"{fp['disk_footprint']['llama3b_dir_gb']} GB"],
                ["LLM narrative", f"{fp['llm_generate']['wall_seconds']} s",
                 f"{fp['llm_generate']['rss_delta_mb']} MB", "--"],
            ]
        rep.table("Table 10(b): client-side pipeline cost (machine-dependent)",
                  ["Step", "Time", "Mem (RSS delta)", "Disk"], rows)

    # Table 10(c) -- client-count scaling
    sc = load(os.path.join(R, "scaling.json"))
    if sc:
        rows = []
        for C in sorted({r["n_clients"] for r in sc}):
            row = [C]
            cell = {}
            for part in ("dirichlet", "iid"):
                rs = [r for r in sc if r["n_clients"] == C and r["partition"] == part and r["status"] == "ok"]
                cell[part] = {
                    "rows": mean([r["mean_n_train"] for r in rs]),
                    "acc": mean([r["accuracy"] for r in rs]),
                    "fid": mean([r["fidelity"] for r in rs]),
                    "usable": mean([r["usable"] for r in rs]),
                    "t": mean([r["mean_client_time_s"] for r in rs]),
                    "n_ok": len(rs),
                }
            row += [fmt(cell["dirichlet"]["rows"], 0), fmt(cell["dirichlet"]["acc"]), fmt(cell["iid"]["acc"]),
                    fmt(cell["dirichlet"]["fid"]), fmt(cell["iid"]["fid"]),
                    f"{fmt(cell['dirichlet']['usable'], 1)}/{C}",
                    f"{fmt(cell['dirichlet']['t'], 1)} / {fmt(cell['iid']['t'], 1)} s",
                    f"{cell['dirichlet']['n_ok']}/{cell['iid']['n_ok']}"]
            rows.append(row)
        rep.table("Table 10(c): client-count scaling on Heart Disease (mean over classifiers)",
                  ["C", "Rows/client", "Acc N-IID", "Acc IID", "Fid N-IID", "Fid IID",
                   "Usable N-IID", "Time/client N-IID / IID", "Runs ok N-IID/IID"], rows)

    out = os.path.join(R, "tables.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Result tables\n\nGenerated by scripts/make_tables.py from " +
                os.path.relpath(R, config.REPO_ROOT) + "/\n\n" + "\n".join(rep.lines))
    print(f"\nWritten -> {out}")


if __name__ == "__main__":
    main()
