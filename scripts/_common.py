"""Shared helpers for the scripts in this folder."""
import argparse
import os
import sys
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fedneuroxai import config  # noqa: E402

# Reruns go here by default so the shipped paper results in results/ are
# never overwritten.
RERUN_DIR = os.path.join(REPO_ROOT, "results_rerun")

# sklearn warns about one-row client shards on small non-IID splits; these
# shards are expected and handled (see partition.make_client_splits).
warnings.filterwarnings("ignore", message="Only one sample available")


def base_parser(description):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(config.DATASETS),
                   choices=list(config.DATASETS))
    p.add_argument("--models", nargs="+", default=config.MODEL_KEYS, choices=config.MODEL_KEYS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", default=RERUN_DIR, help="output root (default: results_rerun/)")
    return p


def config_bins_epochs(dataset_key):
    d = config.DATASETS[dataset_key]
    return d["n_bins"], d["epochs"]
