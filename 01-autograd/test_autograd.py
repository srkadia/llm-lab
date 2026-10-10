"""
checks autograd.py against finite differences.
python test_autograd.py   (pytest works too)
"""

import sys
import numpy as np

from autograd import Tensor, softmax, cross_entropy
from gradcheck import gradcheck, TOL

rng = np.random.default_rng(42)

def rand(*shape):
    return rng.standard_normal(shape)

def rand_pos(*shape):
    # for log / division, keep things away from 0
    return np.abs(rng.standard_normal(shape)) + 0.5

# (name, f, inputs). mostly small shapes, the point is the broadcasting / batching cases
OPS = [
    ("add", lambda t: t[0] + t[1], [rand(3, 4), rand(3, 4)]),
    ("add bias (3,4)+(4,)", lambda t: t[0] + t[1], [rand(3, 4), rand(4)]),
    ("add (3,1)+(1,4)", lambda t: t[0] + t[1], [rand(3, 1), rand(1, 4)]),
    ("mul (3,4)*(3,1)", lambda t: t[0] * t[1], [rand(3, 4), rand(3, 1)]),
    ("div", lambda t: t[0] / t[1], [rand(3, 4), rand_pos(3, 4)]),
    ("pow", lambda t: t[0]**3, [rand(3, 4)]),
    ("exp", lambda t: t[0].exp(), [rand(3, 4)]),
    ("log", lambda t: t[0].log(), [rand_pos(3, 4)]),
    ("tanh", lambda t: t[0].tanh(), [rand(3, 4)]),
    ("relu", lambda t: t[0].relu(), [rand(3, 4)]),
    ("x*x + x (reuse)", lambda t: t[0] * t[0] + t[0], [rand(3, 4)]),
    ("matmul", lambda t: t[0] @ t[1], [rand(3, 4), rand(4, 5)]),
    ("matmul batched (2,3,4)@(4,5)", lambda t: t[0] @ t[1], [rand(2, 3, 4), rand(4, 5)]),
    ("matmul batched (2,3,4)@(2,4,5)", lambda t: t[0] @ t[1], [rand(2, 3, 4), rand(2, 4, 5)]),
    ("sum", lambda t: t[0].sum(), [rand(3, 4)]),
    ("sum axis=0", lambda t: t[0].sum(axis=0), [rand(3, 4)]),
    ("sum axis=1 keepdims", lambda t: t[0].sum(axis=1, keepdims=True), [rand(3, 4)]),
    ("mean axis=-1", lambda t: t[0].mean(axis=-1), [rand(3, 4)]),
    ("reshape", lambda t: t[0].reshape(4, 3), [rand(3, 4)]),
    ("transpose (2,0,1)", lambda t: t[0].transpose((2, 0, 1)), [rand(2, 3, 4)]),
    ("tanh(x @ W + b)", lambda t: (t[0] @ t[1] + t[2]).tanh(), [rand(5, 3), rand(3, 4), rand(4)]),
    ("softmax", lambda t: softmax(t[0]), [rand(6, 5)]),
]


def test_ops():
    for name, f, inputs in OPS:
        err = gradcheck(f, inputs)
        assert err < TOL, f"{name}: {err:.1e}"

def test_cross_entropy():
    y = np.array([0, 3, 1, 4, 2, 3])
    err = gradcheck(lambda t: cross_entropy(t[0], y), [rand(6, 5)])
    assert err < TOL, f"{err:.1e}"

def test_cross_entropy_closed_form():
    # the engine never sees this formula, it should still land on it: (softmax - onehot) / N
    x, y = rand(6, 5), np.array([0, 3, 1, 4, 2, 3])
    logits = Tensor(x)
    cross_entropy(logits, y).backward()
    expected = (softmax(Tensor(x)).data - np.eye(5)[y]) / len(y)
    assert np.abs(logits.grad - expected).max() < 1e-12

def test_softmax_big_logits():
    # naive softmax gives nan here
    p = softmax(Tensor([[1000.0, 1001.0, 1002.0]])).data
    assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-12

def test_mlp():
    # whole 2 layer net at once, all 4 params
    X, y = rand(8, 2), np.array([0, 1, 2, 0, 1, 2, 0, 1])
    def loss(t):
        W1, b1, W2, b2 = t
        h = (Tensor(X) @ W1 + b1).tanh()
        return cross_entropy(h @ W2 + b2, y)
    err = gradcheck(loss, [rand(2, 16) * 0.5, np.zeros(16), rand(16, 3) * 0.5, np.zeros(3)])
    assert err < TOL, f"{err:.1e}"

def test_backward_twice_accumulates():
    # this is why you zero the grads every step
    x = Tensor([1.5])
    y = x * x
    y.backward()
    y.backward()
    assert np.allclose(x.grad, 6.0)

def test_deep_graph():
    # used to die with RecursionError past ~1000 ops. a real model gets that deep quickly
    x = Tensor(np.ones(3))
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    assert np.allclose(x.grad, 1.0001**5000)

def test_gradcheck_catches_bug():
    # sanity check the checker itself. if this passes a wrong grad, every test above is meaningless
    def bad_square(t):
        out = Tensor(t.data**2, (t,), 'bad_square')
        def _backward():
            t.grad += 1.9 * t.data * out.grad # should be 2
        out._backward = _backward
        return out
    err = gradcheck(lambda t: bad_square(t[0]), [rand(3, 4)])
    assert err > 1e-3, f"didn't catch it, {err:.1e}"


if __name__ == '__main__':
    fails = 0
    for name, f, inputs in OPS:
        err = gradcheck(f, inputs)
        ok = err < TOL
        fails += not ok
        print(f"{name:32s} {err:.1e}  {'ok' if ok else 'FAIL'}")
    print()
    others = [test_cross_entropy, test_cross_entropy_closed_form, test_softmax_big_logits,
              test_mlp, test_backward_twice_accumulates, test_deep_graph, test_gradcheck_catches_bug]
    for t in others:
        try:
            t()
            print(f"{t.__name__:32s} ok")
        except AssertionError as e:
            fails += 1
            print(f"{t.__name__:32s} FAIL {e}")
    n = len(OPS) + len(others)
    print(f"\n{n - fails}/{n} passed")
    sys.exit(1 if fails else 0)
