import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from knn import knn_predict


def test_tiny_one_dimensional_example():
    X_tr = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_tr = np.array([0, 0, 1, 1])
    X_te = np.array([[0.4], [10.6], [4.0], [7.0]])
    np.testing.assert_array_equal(knn_predict(X_tr, y_tr, X_te, 1), [0, 1, 0, 1])


def test_majority_vote_with_k_3():
    X_tr = np.array([[0.0], [0.1], [0.2], [5.0], [5.1]])
    y_tr = np.array([0, 0, 1, 1, 1])
    # nearest three to 0.05 are labels 0, 0, 1 -> 0
    assert knn_predict(X_tr, y_tr, np.array([[0.05]]), 3)[0] == 0


def test_k_1_reproduces_training_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = rng.integers(0, 3, 40)
    np.testing.assert_array_equal(knn_predict(X, y, X, 1), y)


def test_output_shape_and_dtype():
    rng = np.random.default_rng(1)
    out = knn_predict(rng.normal(size=(30, 2)), rng.integers(0, 2, 30), rng.normal(size=(7, 2)), 5)
    assert out.shape == (7,)
    assert np.issubdtype(out.dtype, np.integer)


def test_matches_sklearn_on_wine():
    X, y = load_wine(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    A, B = (X_tr - mu) / sd, (X_te - mu) / sd
    ref = KNeighborsClassifier(5).fit(A, y_tr).predict(B)
    mine = knn_predict(A, y_tr, B, 5)
    assert np.mean(mine == ref) > 0.98
    assert np.mean(mine == y_te) > 0.9
