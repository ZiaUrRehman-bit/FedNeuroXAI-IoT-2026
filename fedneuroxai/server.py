"""
Server-side code: aggregation of what the clients send (Section 3.2).

  - Shared feature scaler: combines client (n, mean, variance) with the
    parallel mean/variance formula (Eq. 3-4, without class conditioning).
  - FedAvg for logistic regression, linear SVM and MLP: sample-weighted
    average of client weights each round (Eq. 2), per layer for the MLP.
  - Exact sufficient-statistic aggregation for Gaussian Naive Bayes and LDA:
    reconstructs the model a single fit on the pooled data would give.

The server never sees raw rows. federated_train() simulates the full
client/server loop in one process.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from . import client as fl_client

SUFFICIENT_STAT_MODELS = {"gaussian_nb", "lda"}


# ---------------------------------------------------------------------------
# Shared feature scaler
# ---------------------------------------------------------------------------

class FederatedScaler:
    """StandardScaler-compatible transform built from aggregated client
    statistics. Every client and the evaluator apply the same scaling, which
    FedAvg weight averaging requires."""

    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X):
        return (np.asarray(X) - self.mean_) / self.scale_


def aggregate_scaler(client_stats) -> FederatedScaler:
    """client_stats: list of (n, mean, var) from client.local_feature_stats."""
    ns = np.array([s[0] for s in client_stats], dtype=float)
    means = [s[1] for s in client_stats]
    vars_ = [s[2] for s in client_stats]
    overall_mean = np.average(means, axis=0, weights=ns)
    overall_var = np.average(
        [v + (mu - overall_mean) ** 2 for v, mu in zip(vars_, means)],
        axis=0, weights=ns,
    )
    scale = np.sqrt(overall_var)
    scale[scale == 0] = 1.0
    return FederatedScaler(overall_mean, scale)


def fit_federated_scaler(clients) -> FederatedScaler:
    """Convenience wrapper: each client computes its stats, server combines."""
    return aggregate_scaler([fl_client.local_feature_stats(c["X_train"]) for c in clients])


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def average_linear_params(client_models, weights):
    coef = np.average([m.coef_ for m in client_models], axis=0, weights=weights)
    intercept = np.average([m.intercept_ for m in client_models], axis=0, weights=weights)
    return coef, intercept


def average_mlp_params(client_models, weights):
    n_layers = len(client_models[0].coefs_)
    coefs = [np.average([m.coefs_[l] for m in client_models], axis=0, weights=weights)
             for l in range(n_layers)]
    intercepts = [np.average([m.intercepts_[l] for m in client_models], axis=0, weights=weights)
                  for l in range(n_layers)]
    return coefs, intercepts


def fedavg_train(model_key, client_Xs, client_ys, classes, n_rounds, seed):
    weights = np.array([len(y) for y in client_ys], dtype=float)
    client_models = [fl_client.build_local_model(model_key, seed) for _ in client_Xs]

    for rnd in range(n_rounds):
        # 1. each client trains locally on its own shard
        for m, Xc, yc in zip(client_models, client_Xs, client_ys):
            fl_client.local_update(m, Xc, yc, classes, first_round=(rnd == 0))
        # 2. server averages the weights and broadcasts them back
        if model_key == "mlp":
            coefs, intercepts = average_mlp_params(client_models, weights)
            for m in client_models:
                m.coefs_ = [c.copy() for c in coefs]
                m.intercepts_ = [b.copy() for b in intercepts]
        else:
            coef, intercept = average_linear_params(client_models, weights)
            for m in client_models:
                m.coef_ = coef.copy()
                m.intercept_ = intercept.copy()

    return client_models[0]


# ---------------------------------------------------------------------------
# Exact sufficient-statistic aggregation
# ---------------------------------------------------------------------------

def combine_mean_var(ns, means, vars_):
    """Parallel combination of per-group (n, mean, var) -> pooled (mean, var)."""
    ns = np.asarray(ns, dtype=float)
    overall_mean = np.average(means, axis=0, weights=ns)
    overall_var = np.average(
        [v + (mu - overall_mean) ** 2 for v, mu in zip(vars_, means)],
        axis=0, weights=ns,
    )
    return overall_mean, overall_var


def combine_mean_scatter(ns, means, scatters):
    """Pairwise parallel combination of per-group (n, mean, scatter matrix)."""
    n_acc, mean_acc, scat_acc = ns[0], means[0], scatters[0]
    for n_g, mean_g, scat_g in zip(ns[1:], means[1:], scatters[1:]):
        n_total = n_acc + n_g
        delta = mean_g - mean_acc
        mean_acc = mean_acc + delta * (n_g / n_total)
        scat_acc = scat_acc + scat_g + np.outer(delta, delta) * (n_acc * n_g / n_total)
        n_acc = n_total
    return mean_acc, scat_acc, n_acc


def aggregate_gaussian_nb(client_models, classes, n_features):
    n_classes = len(classes)
    agg_theta = np.zeros((n_classes, n_features))
    agg_var = np.zeros((n_classes, n_features))
    agg_count = np.zeros(n_classes)

    for c_idx, c in enumerate(classes):
        ns, means, vars_ = [], [], []
        for m in client_models:
            if c in m.classes_:
                li = list(m.classes_).index(c)
                ns.append(float(m.class_count_[li]))
                means.append(m.theta_[li])
                vars_.append(m.var_[li])
        if not ns:
            continue
        mean, var = combine_mean_var(ns, means, vars_)
        agg_theta[c_idx], agg_var[c_idx], agg_count[c_idx] = mean, var, sum(ns)

    g = GaussianNB()
    g.classes_ = classes
    g.theta_ = agg_theta
    g.var_ = agg_var
    g.class_count_ = agg_count
    g.class_prior_ = agg_count / agg_count.sum()
    g.epsilon_ = float(np.mean([m.epsilon_ for m in client_models]))
    g.n_features_in_ = n_features
    return g


def aggregate_lda(client_models, classes, n_features):
    n_classes = len(classes)
    class_means = np.zeros((n_classes, n_features))
    class_counts = np.zeros(n_classes)
    pooled_scatter = np.zeros((n_features, n_features))
    n_total = 0.0

    for c_idx, c in enumerate(classes):
        ns, means, scatters = [], [], []
        for m in client_models:
            if c in m.classes_:
                li = list(m.classes_).index(c)
                ns.append(m._class_n[li])
                means.append(m._class_mean[li])
                scatters.append(m._class_scatter[li])
        if not ns:
            continue
        mean, scatter, n_c = combine_mean_scatter(ns, means, scatters)
        class_means[c_idx] = mean
        class_counts[c_idx] = n_c
        pooled_scatter += scatter
        n_total += n_c

    priors = class_counts / n_total
    cov = pooled_scatter / (n_total - n_classes)      # shared within-class covariance
    cov_inv = np.linalg.pinv(cov)
    coef = class_means @ cov_inv
    intercept = -0.5 * np.einsum("cf,cf->c", class_means, coef) + np.log(priors)

    g = LinearDiscriminantAnalysis()
    g.classes_ = classes
    g.priors_ = priors
    g.means_ = class_means
    g.covariance_ = cov
    g.n_features_in_ = n_features
    if n_classes == 2:
        # sklearn's binary convention: one row scoring classes_[1] vs classes_[0]
        g.coef_ = (coef[1] - coef[0]).reshape(1, -1)
        g.intercept_ = np.array([intercept[1] - intercept[0]])
    else:
        g.coef_ = coef
        g.intercept_ = intercept
    return g


def suffstat_train(model_key, client_Xs, client_ys, classes):
    n_features = client_Xs[0].shape[1]
    if model_key == "gaussian_nb":
        local = [fl_client.local_gaussian_nb(Xc, yc) for Xc, yc in zip(client_Xs, client_ys)]
        return aggregate_gaussian_nb(local, classes, n_features)
    if model_key == "lda":
        local = [fl_client.local_lda_stats(Xc, yc) for Xc, yc in zip(client_Xs, client_ys)]
        return aggregate_lda(local, classes, n_features)
    raise ValueError(model_key)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def federated_train(model_key, client_Xs, client_ys, n_rounds=100, seed=42):
    """Train one global model across clients.

    client_Xs, client_ys: per-client numpy arrays, already transformed with
    the shared federated scaler. Returns a fitted sklearn classifier.
    """
    classes = np.unique(np.concatenate(client_ys))
    if model_key in SUFFICIENT_STAT_MODELS:
        return suffstat_train(model_key, client_Xs, client_ys, classes)
    return fedavg_train(model_key, client_Xs, client_ys, classes, n_rounds, seed)
