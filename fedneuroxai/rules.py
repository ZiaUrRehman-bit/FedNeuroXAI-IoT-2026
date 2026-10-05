"""
Neuro-symbolic rule mining (PyAerial) and rule-set evaluation
(Section 3.4: Eqs. 5-8).
"""
from typing import Dict

import numpy as np
import pandas as pd
from aerial import model as aerial_model
from aerial import rule_extraction

RULE_COLUMNS = ["antecedents", "consequent", "support", "confidence", "coverage", "zhang"]


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def extract_and_filter(disc_train_df: pd.DataFrame, epochs: int = 10,
                       min_support: float = 0.10, min_confidence: float = 0.80,
                       min_zhang: float = 0.30):
    """Train the PyAerial autoencoder on the discretised, surrogate-labelled
    table, mine association rules, and keep those passing all thresholds.
    Returns (df_rules, aerial_statistics)."""
    autoenc = aerial_model.train(disc_train_df, epochs=epochs)
    rules_dict = rule_extraction.generate_rules(autoenc)
    all_rules, stats = rules_dict["rules"], rules_dict["statistics"]

    kept = [
        r for r in all_rules
        if r["support"] >= min_support
        and r["confidence"] >= min_confidence
        and r["zhangs_metric"] >= min_zhang
    ]
    print(f"[rules] mined {len(all_rules)}, kept {len(kept)} after thresholds")
    return pd.DataFrame([_rule_to_row(r) for r in kept]), stats


def _rule_to_row(r: Dict) -> Dict:
    ants = " AND ".join(f"{a['feature']}={a['value']}" for a in r["antecedents"])
    return {
        "antecedents": ants,
        "consequent": f"{r['consequent']['feature']}={r['consequent']['value']}",
        "support": r["support"],
        "confidence": r["confidence"],
        "coverage": r["rule_coverage"],
        "zhang": r["zhangs_metric"],
    }


def keep_class_rules(df_rules: pd.DataFrame, class_feature: str = "class") -> pd.DataFrame:
    """Keep only rules whose consequent is the (surrogate) class label."""
    if df_rules.empty or "consequent" not in df_rules.columns:
        return pd.DataFrame(columns=RULE_COLUMNS)
    mask = df_rules["consequent"].str.startswith(class_feature + "=")
    return df_rules[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prediction with a rule set (Eq. 5)
# ---------------------------------------------------------------------------

def rule_mask(antecedent_str: str, df_disc: pd.DataFrame) -> np.ndarray:
    """Boolean mask of rows on which every antecedent condition holds."""
    mask = np.ones(len(df_disc), dtype=bool)
    for part in (p.strip() for p in antecedent_str.split("AND")):
        feat, val = part.split("=", 1)
        mask &= (df_disc[feat.strip()].astype(str) == val.strip())
    return mask


def ruleset_predict(df_rules: pd.DataFrame, df_disc: pd.DataFrame) -> np.ndarray:
    """Highest-confidence firing rule wins; -1 where no rule fires."""
    n = len(df_disc)
    y_pred = np.full(n, -1, dtype=int)
    conf_used = np.full(n, -1.0)
    for _, row in df_rules.iterrows():
        mask = rule_mask(row["antecedents"], df_disc)
        cons_val = int(row["consequent"].split("=")[1].strip())
        conf = float(row["confidence"])
        update = mask & (conf > conf_used)
        y_pred[update] = cons_val
        conf_used[update] = conf
    return y_pred


# ---------------------------------------------------------------------------
# Rule-set metrics
# ---------------------------------------------------------------------------

def average_rule_length(df_rules: pd.DataFrame) -> float:
    if df_rules.empty:
        return 0.0
    lengths = df_rules["antecedents"].astype(str).apply(
        lambda s: 1 + s.count("AND") if s.strip() else 0).astype(float)
    return float(lengths.mean())


def class_coverage(df_rules: pd.DataFrame, all_classes=(0, 1)) -> float:
    """Fraction of classes predicted by at least one rule."""
    if df_rules.empty:
        return 0.0

    def extract_class(consequent):
        s = str(consequent).strip()
        if not s.startswith("class="):
            return None
        try:
            return int(s.split("=", 1)[1].strip())
        except (ValueError, IndexError):
            return None

    cons_vals = df_rules["consequent"].astype(str).apply(extract_class).dropna().astype(int)
    if cons_vals.empty:
        return 0.0
    return len(set(cons_vals.unique()) & set(all_classes)) / len(all_classes)


def fraction_overlap(df_rules: pd.DataFrame, df_disc: pd.DataFrame) -> float:
    """Mean pairwise fraction of instances covered by two rules ("Ovl")."""
    R = len(df_rules)
    if R <= 1:
        return 0.0
    masks = [rule_mask(row["antecedents"], df_disc) for _, row in df_rules.iterrows()]
    N = len(df_disc)
    total = sum(np.sum(masks[i] & masks[j]) / N for i in range(R) for j in range(i + 1, R))
    return float((2 / (R * (R - 1))) * total)


def evaluate(df_rules: pd.DataFrame, df_disc_test: pd.DataFrame, y_bb_test: np.ndarray,
             all_classes=(0, 1)) -> dict:
    """All per-client rule metrics on the client's local test split.

    completeness = coverage (Eq. 7); fidelity = agreement with the global
    model, counting non-firing instances as disagreement (Eq. 6).
    """
    y_true = df_disc_test["class"].astype(int).values
    y_pred = ruleset_predict(df_rules, df_disc_test)
    return {
        "completeness": float(np.mean(y_pred != -1)),
        "correctness": float(np.mean(y_pred == y_true)),
        "fidelity": float(np.mean(y_pred == y_bb_test)),
        "num_rules": len(df_rules),
        "avg_length": average_rule_length(df_rules),
        "class_coverage": class_coverage(df_rules, all_classes),
        "overlap": fraction_overlap(df_rules, df_disc_test),
        "avg_support": float(df_rules["support"].mean()),
        "avg_confidence": float(df_rules["confidence"].mean()),
        "avg_zhang": float(df_rules["zhang"].mean()),
    }
