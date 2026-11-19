import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from digits_net import DigitsNet, accuracy, train  # noqa: E402


def split():
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def test_output_shape_and_parameter_count():
    net = DigitsNet(hidden=32)
    out = net(torch.zeros(5, 64))
    assert out.shape == (5, 10)
    assert sum(p.numel() for p in net.parameters()) == 64 * 32 + 32 + 32 * 10 + 10


def test_hidden_size_is_respected():
    assert sum(p.numel() for p in DigitsNet(hidden=8).parameters()) == 64 * 8 + 8 + 8 * 10 + 10


def test_outputs_are_logits_not_probabilities():
    net = DigitsNet()
    out = net(torch.randn(20, 64) * 5)
    assert not torch.allclose(out.sum(1), torch.ones(20), atol=1e-3)


def test_accuracy_is_a_python_float_in_range():
    X_tr, X_te, y_tr, y_te = split()
    acc = accuracy(DigitsNet(), X_te, y_te)
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0


def test_train_returns_epoch_losses_and_learns():
    X_tr, X_te, y_tr, y_te = split()
    torch.manual_seed(0)
    net = DigitsNet(hidden=32)
    losses = train(net, X_tr, y_tr, epochs=40, lr=0.1)
    assert len(losses) == 40
    assert losses[-1] < losses[0] / 5
    assert accuracy(net, X_te, y_te) > 0.95


def test_training_is_reproducible_given_seeds():
    X_tr, X_te, y_tr, y_te = split()
    results = []
    for _ in range(2):
        torch.manual_seed(0)
        net = DigitsNet(hidden=16)
        results.append(train(net, X_tr, y_tr, epochs=3, lr=0.1, seed=5))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6)
