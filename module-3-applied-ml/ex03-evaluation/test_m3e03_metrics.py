import numpy as np
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.metrics import precision_recall_fscore_support

from metrics import confusion_matrix, kfold_indices, precision_recall_f1


def test_confusion_matrix_small_example():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 2]])


def test_confusion_matrix_matches_sklearn_and_keeps_empty_classes():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 100)
    y_pred = rng.integers(0, 4, 100)
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred, 5)[:4, :4], sk_cm(y_true, y_pred, labels=range(4)))
    assert confusion_matrix(y_true, y_pred, 5).shape == (5, 5)


def test_prf_matches_sklearn():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 5, 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, rng.integers(0, 5, 300))
    p, r, f = precision_recall_f1(confusion_matrix(y_true, y_pred, 5))
    rp, rr, rf, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    np.testing.assert_allclose(p, rp, atol=1e-12)
    np.testing.assert_allclose(r, rr, atol=1e-12)
    np.testing.assert_allclose(f, rf, atol=1e-12)


def test_prf_handles_a_class_that_is_never_predicted():
    cm = np.array([[5, 0], [3, 0]])  # class 1 never predicted
    p, r, f = precision_recall_f1(cm)
    assert p[1] == 0.0 and r[1] == 0.0 and f[1] == 0.0
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(f))


def test_kfold_covers_every_index_once_with_balanced_folds():
    n, k = 103, 5
    folds = list(kfold_indices(n, k, seed=3))
    assert len(folds) == k
    all_val = np.concatenate([v for _, v in folds])
    assert sorted(all_val.tolist()) == list(range(n))
    sizes = [len(v) for _, v in folds]
    assert max(sizes) - min(sizes) <= 1
    for tr, va in folds:
        assert set(tr).isdisjoint(va)
        assert len(tr) + len(va) == n


def test_kfold_is_seeded_and_shuffled():
    a = [v.tolist() for _, v in kfold_indices(50, 5, seed=1)]
    b = [v.tolist() for _, v in kfold_indices(50, 5, seed=1)]
    c = [v.tolist() for _, v in kfold_indices(50, 5, seed=2)]
    assert a == b and a != c
    assert a[0] != list(range(10))  # not just consecutive blocks
