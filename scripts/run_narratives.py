"""
LLM narrative faithfulness (Table 8). For each (dataset, classifier) under
non-IID federation, the first client with a non-empty rule set gets one
narrative from the local LLM; every numeric threshold in the narrative is
checked against the rules the LLM was given.

Requires: pip install -r requirements-llm.txt, plus access to
meta-llama/Llama-3.2-3B-Instruct (or pass --llm /path/to/local/model).
CPU, fp32, greedy decoding: roughly 4-5 minutes per narrative, ~13 GB RAM.

    python scripts/run_narratives.py
    python scripts/run_narratives.py --llm C:/models/Llama-3.2-3B-Instruct

Output: <out>/narrative_faithfulness.json
"""
import json
import os
import time

from _common import base_parser, config_bins_epochs
from fedneuroxai import config, narrative
from fedneuroxai.client import explain_client
from fedneuroxai.experiment import build_federation, set_seed


def main():
    p = base_parser(__doc__)
    p.add_argument("--llm", default=config.LLM_PATH, help="HF model id or local directory")
    args = p.parse_args()

    # 1. mine rules (cheap) for every combination first
    chosen = {}
    for ds in args.datasets:
        n_bins, epochs = config_bins_epochs(ds)
        for m in args.models:
            set_seed(args.seed)
            _, clients, _, _, scaler, model = build_federation(ds, m, seed=args.seed)
            chosen[(ds, m)] = None
            for cid, c in enumerate(clients):
                exp = explain_client(c, model, scaler, epochs=epochs, n_bins=n_bins,
                                     min_support=config.MIN_SUPPORT,
                                     min_confidence=config.MIN_CONFIDENCE,
                                     min_zhang=config.MIN_ZHANG)
                if len(exp["rules"]) > 0:
                    chosen[(ds, m)] = (cid, exp["rules"])
                    break

    # 2. one narrative per combination
    tok, pipe = narrative.load_llm(args.llm)
    per_client = []
    for (ds, m), pick in chosen.items():
        if pick is None:
            per_client.append({"dataset": ds, "model": m, "client": None, "skipped": True})
            continue
        cid, df_rules = pick
        sys_msg, usr_msg, rules_text = narrative.build_prompt(df_rules)
        t0 = time.perf_counter()
        text = narrative.generate_narrative(tok, pipe, sys_msg, usr_msg)
        r = {"dataset": ds, "model": m, "client": cid,
             "wall_seconds": round(time.perf_counter() - t0, 1), "narrative": text}
        r.update(narrative.threshold_faithfulness(text, rules_text))
        per_client.append(r)
        print(f"[{ds}/{m}] client {cid}: {r['n_matched_verbatim']}/{r['n_thresholds_in_narrative']} "
              f"thresholds matched, {r['wall_seconds']} s")

    valid = [r for r in per_client if not r.get("skipped") and r["n_thresholds_in_narrative"]]
    total = sum(r["n_thresholds_in_narrative"] for r in valid)
    matched = sum(r["n_matched_verbatim"] for r in valid)
    summary = {
        "n_narratives_generated": len(valid),
        "n_skipped_no_rules": sum(1 for r in per_client if r.get("skipped")),
        "total_thresholds_checked": total,
        "total_matched_verbatim": matched,
        "overall_faithfulness_rate": round(matched / total, 4) if total else None,
        "n_narratives_zero_hallucinations": sum(1 for r in valid if r["n_unmatched"] == 0),
    }
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "narrative_faithfulness.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_client": per_client}, f, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
