"""Plumbing for exercise 2.5: data splits and a configurable sklearn MLP."""

import warnings

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=Warning, module="sklearn")


def load_split(n_train=None, seed=0):
    """Digits scaled to [0, 1], split into train / validation / test (60/20/20).

    If n_train is given, the training set is cut down to that many samples
    (validation and test stay the same size, so scores remain comparable).
    """
    d = load_digits()
    X, y = d.data / 16.0, d.target
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    if n_train is not None:
        X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def make_mlp(**overrides):
    """Plain minibatch SGD (no momentum) so the knobs mean what the textbook says.

    Any MLPClassifier argument can be overridden, e.g. make_mlp(learning_rate_init=0.01, alpha=1e-3).
    """
    cfg = dict(
        hidden_layer_sizes=(32,),
        solver="sgd",
        momentum=0.0,
        learning_rate_init=0.1,
        batch_size=32,
        max_iter=50,
        tol=0,
        n_iter_no_change=10_000,  # never stop early unless you ask
        random_state=0,
    )
    cfg.update(overrides)
    return MLPClassifier(**cfg)


def run(split=None, **overrides):
    """Fit one configuration; return a dict with the fitted model and its scores."""
    X_tr, X_val, X_te, y_tr, y_val, y_te = split or load_split()
    model = make_mlp(**overrides).fit(X_tr, y_tr)
    return {
        "model": model,
        "train_acc": model.score(X_tr, y_tr),
        "val_acc": model.score(X_val, y_val),
        "loss": model.loss_,
        "loss_curve": model.loss_curve_,
    }
