import numpy as np
import pytest
from scipy.signal import correlate2d

from conv import conv2d, max_pool2d


def test_conv_matches_scipy_cross_correlation():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(8, 8))
    k = rng.normal(size=(3, 3))
    out = conv2d(img, k)
    assert out.shape == (6, 6)
    np.testing.assert_allclose(out, correlate2d(img, k, mode="valid"), atol=1e-10)


def test_conv_non_square_image_and_kernel():
    rng = np.random.default_rng(1)
    img = rng.normal(size=(6, 9))
    k = rng.normal(size=(2, 4))
    np.testing.assert_allclose(conv2d(img, k), correlate2d(img, k, mode="valid"), atol=1e-10)


def test_conv_does_not_flip_the_kernel():
    img = np.zeros((3, 3))
    img[0, 0] = 1.0
    k = np.array([[1.0, 2.0], [3.0, 4.0]])
    # cross-correlation: the output at (0,0) is img[0,0]*k[0,0] = 1
    assert conv2d(img, k)[0, 0] == 1.0


def test_max_pool_known_example():
    img = np.array([[1, 2, 5, 6], [3, 4, 7, 8], [9, 1, 0, 2], [3, 3, 1, 1]], dtype=float)
    np.testing.assert_array_equal(max_pool2d(img, 2), [[4, 8], [9, 2]])


def test_max_pool_drops_remainder():
    img = np.arange(25, dtype=float).reshape(5, 5)
    out = max_pool2d(img, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == 6 and out[1, 1] == 18


def test_cnn_shapes_and_budget():
    torch = pytest.importorskip("torch")
    from cnn import DigitsCNN

    net = DigitsCNN()
    out = net(torch.zeros(4, 1, 8, 8))
    assert out.shape == (4, 10)
    n = sum(p.numel() for p in net.parameters())
    assert n == (8 * 9 + 8) + (8 * 3 * 3 * 10 + 10)  # the architecture in the README
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters())
