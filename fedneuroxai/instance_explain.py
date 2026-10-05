"""
Per-prediction justification (Section 3.6).

The rule-based local explanation of an instance is the highest-confidence
rule of the client's rule set that fires on it. SHAP and LIME are computed
on the same instance and model, using only the client's own training data
as background, and compared with the fired rule via Jaccard overlap (Eq. 9).
"""
import re

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from .rules import rule_mask

LINEAR_MODEL_KEYS = {"logistic_regression", "linear_svm", "lda"}


def fired_rule_for_instance(df_rules: pd.DataFrame, disc_instance_row: pd.Series):
    """Highest-confidence rule firing on this discretised instance, or None."""
    row_df = disc_instance_row.to_frame().T.reset_index(drop=True)
    best = None
    for _, r in df_rules.iterrows():
        if rule_mask(r["antecedents"], row_df)[0]:
            if best is None or r["confidence"] > best["confidence"]:
                best = r
    return best


def shap_top_features(model, model_key, X_train_scaled, X_instance_scaled, feature_names):
    """|SHAP| per feature for one instance: LinearExplainer for linear
    models, permutation-based Explainer otherwise; 50-row local background."""
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train_scaled), size=min(50, len(X_train_scaled)), replace=False)
    background = X_train_scaled[bg_idx]
    instance_2d = np.asarray(X_instance_scaled).reshape(1, -1)

    if model_key in LINEAR_MODEL_KEYS:
        sv = shap.LinearExplainer(model, background).shap_values(instance_2d)
    else:
        sv = shap.Explainer(model.predict_proba, background)(instance_2d).values

    sv = np.asarray(sv).reshape(-1, len(feature_names))
    return pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False)


def _lime_condition_to_feature(condition: str, feature_names) -> str:
    """Map a LIME condition such as '-0.62 < plas <= -0.05' to 'plas'."""
    for feat in feature_names:
        if re.search(rf"\b{re.escape(feat)}\b", condition):
            return feat
    return condition


def lime_top_features(model, X_train_scaled, X_instance_scaled, feature_names, class_names,
                      n_features=5):
    """LIME with its default settings (quartile discretisation of continuous
    features), local training data only."""
    explainer = LimeTabularExplainer(
        X_train_scaled, feature_names=list(feature_names),
        class_names=[str(c) for c in class_names], discretize_continuous=True,
    )
    instance_1d = np.asarray(X_instance_scaled).reshape(-1)
    pred_class = int(model.predict(instance_1d.reshape(1, -1))[0])
    exp = explainer.explain_instance(
        instance_1d, model.predict_proba, num_features=n_features, labels=(pred_class,),
    )
    weights = {}
    for condition, w in exp.as_list(label=pred_class):
        feat = _lime_condition_to_feature(condition, feature_names)
        weights[feat] = weights.get(feat, 0.0) + abs(w)
    return pd.Series(weights).sort_values(ascending=False)


def rule_feature_overlap(rule_antecedent_str: str, ranked_features: list, top_k: int = 3) -> float:
    """Jaccard overlap between the fired rule's features and a method's top-k (Eq. 9)."""
    rule_feats = {p.split("=")[0].strip() for p in str(rule_antecedent_str).split("AND")}
    top_feats = set(ranked_features[:top_k])
    if not rule_feats or not top_feats:
        return 0.0
    return len(rule_feats & top_feats) / len(rule_feats | top_feats)


def explain_instance(model, model_key, df_rules, disc_instance_row, X_instance_scaled,
                     X_train_scaled, feature_names, class_names, top_k=3):
    """Fired rule, SHAP and LIME for one instance, plus rule/SHAP and
    rule/LIME overlap. A SHAP or LIME exception is recorded as data (the
    rule-based explanation has no such failure mode), not raised."""
    rule = fired_rule_for_instance(df_rules, disc_instance_row)

    shap_ranking = lime_ranking = None
    shap_error = lime_error = None
    try:
        shap_ranking = shap_top_features(model, model_key, X_train_scaled, X_instance_scaled,
                                         feature_names)
    except Exception as e:
        shap_error = str(e)
    try:
        lime_ranking = lime_top_features(model, X_train_scaled, X_instance_scaled,
                                         feature_names, class_names)
    except Exception as e:
        lime_error = str(e)

    result = {
        "fired_rule": None if rule is None else {
            "antecedents": rule["antecedents"], "consequent": rule["consequent"],
            "confidence": float(rule["confidence"]), "support": float(rule["support"]),
        },
        "shap_ranking": shap_ranking.to_dict() if shap_ranking is not None else None,
        "lime_ranking": lime_ranking.to_dict() if lime_ranking is not None else None,
        "shap_error": shap_error,
        "lime_error": lime_error,
        "rule_shap_overlap": None,
        "rule_lime_overlap": None,
    }
    if rule is not None and shap_ranking is not None:
        result["rule_shap_overlap"] = rule_feature_overlap(
            rule["antecedents"], list(shap_ranking.index), top_k=top_k)
    if rule is not None and lime_ranking is not None:
        result["rule_lime_overlap"] = rule_feature_overlap(
            rule["antecedents"], list(lime_ranking.index), top_k=top_k)
    return result
