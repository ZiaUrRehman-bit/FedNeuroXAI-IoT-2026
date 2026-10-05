"""
Client-side resource footprint (Table 10(b)): wall-clock time, process
memory (RSS) and disk for one representative client -- WBC, client 0, IID
partition, Gaussian Naive Bayes global model (the worked example of
Section 3.4). Federated training rounds are not included.

  1. rule extraction (discretisation + PyAerial + fidelity evaluation)
  2. LLM load and one narrative (skip with --no-llm)
  3. on-disk size of the LLM weights and of the mined rule set

The paper's numbers were measured on an Intel Core Ultra 5 235U laptop
(12 cores, 32 GB RAM, no GPU, Windows 11); yours will differ.

    python scripts/profile_footprint.py
    python scripts/profile_footprint.py --no-llm

Output: <out>/footprint.json
"""
import json
import os
import time

import psutil

from _common import base_parser, config_bins_epochs
from fedneuroxai import config, narrative
from fedneuroxai.client import explain_client
from fedneuroxai.experiment import build_federation, set_seed

PROC = psutil.Process(os.getpid())


def rss_mb():
    return PROC.memory_info().rss / (1024 ** 2)


def dir_size_gb(path):
    if not os.path.isdir(path):
        return None
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1024 ** 3


def main():
    p = base_parser(__doc__)
    p.add_argument("--llm", default=config.LLM_PATH)
    p.add_argument("--no-llm", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    _, clients, _, _, scaler, model = build_federation("wbc", "gaussian_nb", seed=args.seed,
                                                       partition="iid")
    client0 = clients[0]
    n_bins, epochs = config_bins_epochs("wbc")
    results = {}

    base = rss_mb()
    t0 = time.perf_counter()
    exp = explain_client(client0, model, scaler, epochs=epochs, n_bins=n_bins,
                         min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE,
                         min_zhang=config.MIN_ZHANG)
    t1 = time.perf_counter()
    df_rules = exp["rules"]
    results["rule_extraction"] = {
        "n_rules": len(df_rules), "wall_seconds": round(t1 - t0, 2),
        "rss_delta_mb": round(rss_mb() - base, 1), "rss_after_mb": round(rss_mb(), 1),
    }
    print("Rule extraction:", results["rule_extraction"])

    if not args.no_llm:
        base = rss_mb()
        t0 = time.perf_counter()
        tok, pipe = narrative.load_llm(args.llm)
        t_loaded = time.perf_counter()
        load_rss = rss_mb()
        sys_msg, usr_msg, _ = narrative.build_prompt(df_rules, class_labels={0: "Malignant", 1: "Benign"})
        text = narrative.generate_narrative(tok, pipe, sys_msg, usr_msg)
        t1 = time.perf_counter()
        results["llm_load"] = {"wall_seconds": round(t_loaded - t0, 2),
                               "rss_delta_mb": round(load_rss - base, 1), "rss_after_mb": round(load_rss, 1)}
        results["llm_generate"] = {"wall_seconds": round(t1 - t_loaded, 2),
                                   "rss_delta_mb": round(rss_mb() - load_rss, 1),
                                   "rss_after_mb": round(rss_mb(), 1), "n_new_tokens_requested": 400}
        results["narrative_preview"] = text[:300]
        print("LLM load:", results["llm_load"])
        print("LLM generate:", results["llm_generate"])

    results["disk_footprint"] = {
        "llama3b_dir_gb": None if dir_size_gb(args.llm) is None else round(dir_size_gb(args.llm), 2),
        "mined_ruleset_kb": round(len(df_rules.to_json(orient="records").encode("utf-8")) / 1024, 1),
    }
    print("Disk:", results["disk_footprint"], "(LLM size is reported only for a local --llm directory)")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "footprint.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
