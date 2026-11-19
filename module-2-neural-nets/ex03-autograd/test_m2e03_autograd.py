import math

import pytest

from autograd import MLP, Value


def num_grad(f, args, i, eps=1e-6):
    hi = list(args)
    lo = list(args)
    hi[i] += eps
    lo[i] -= eps
    return (f(*hi) - f(*lo)) / (2 * eps)


def test_forward_and_basic_backward():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    f = a * b + c
    assert f.data == 4.0
    f.backward()
    assert (a.grad, b.grad, c.grad) == (-3.0, 2.0, 1.0)


def test_plain_numbers_mix_with_values():
    a = Value(3.0)
    f = 2 * a + 1 - a / 3
    assert abs(f.data - (2 * 3 + 1 - 1)) < 1e-12
    f.backward()
    assert abs(a.grad - (2 - 1 / 3)) < 1e-12


def test_a_value_used_twice_accumulates():
    a = Value(3.0)
    f = a * a
    f.backward()
    assert a.grad == 6.0


def test_reuse_deep_in_the_graph():
    a = Value(2.0)
    b = a * a
    f = b + b + a
    f.backward()
    assert abs(a.grad - (4 * 2 + 1)) < 1e-12  # f = 2a^2 + a


def test_matches_numerical_gradients_on_a_mixed_expression():
    def f(a, b, c):
        return (a * b - c) ** 2 / (1 + b * b) + a / c

    args = [0.7, -1.3, 2.1]
    vs = [Value(x) for x in args]
    out = f(*vs)
    out.backward()
    for i, v in enumerate(vs):
        assert abs(v.grad - num_grad(f, args, i)) < 1e-5


def test_neg_sub_pow_div():
    a, b = Value(1.5), Value(-0.5)
    out = -(a - b) ** 3 / (a * 2)
    out.backward()

    def f(x, y):
        return -((x - y) ** 3) / (x * 2)

    assert abs(a.grad - num_grad(f, [1.5, -0.5], 0)) < 1e-5
    assert abs(b.grad - num_grad(f, [1.5, -0.5], 1)) < 1e-5


def test_exp_tanh_relu():
    a = Value(0.4)
    out = a.exp() * a.tanh() + (a * -1).relu() + a.relu()
    out.backward()

    def f(x):
        return math.exp(x) * math.tanh(x) + max(0.0, -x) + max(0.0, x)

    assert abs(out.data - f(0.4)) < 1e-12
    assert abs(a.grad - num_grad(f, [0.4], 0)) < 1e-5


def test_relu_has_zero_gradient_for_negative_input():
    a = Value(-2.0)
    a.relu().backward()
    assert a.grad == 0.0


def test_mlp_parameter_count_and_gradient_flow():
    net = MLP([2, 4, 1])
    ps = net.parameters()
    assert len(ps) == (2 * 4 + 4) + (4 * 1 + 1)
    out = net([Value(0.5), Value(-1.0)])
    out = out[0] if isinstance(out, list) else out
    out.backward()
    assert any(abs(p.grad) > 0 for p in ps)


def test_mlp_can_fit_a_single_point():
    net = MLP([2, 3, 1])
    x = [Value(0.5), Value(-1.0)]
    first = None
    for _ in range(100):
        out = net(x)
        out = out[0] if isinstance(out, list) else out
        loss = (out - 0.5) ** 2
        first = loss.data if first is None else first
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.data -= 0.1 * p.grad
    assert loss.data < first / 100
