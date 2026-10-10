import numpy as np

from autograd import Tensor

# correct grads land around 1e-9..1e-11 in float64 (see run.py), so 1e-6 leaves plenty of room
TOL = 1e-6


def rel_error(a, b):
    # norm based, so entries where the true grad is ~0 don't blow it up
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def gradcheck(f, inputs, h=1e-6, seed=0):
    """
    compare the autograd grads of f against central differences, return the worst rel error.
    f takes a list of Tensors and returns a Tensor. inputs is a list of float64 arrays.
    keep this in float64, float32 is too noisy to catch small bugs (run.py shows why)
    """
    # don't just .sum() the output: then every output gets grad 1 and some bugs hide,
    # e.g. forgetting to transpose the grad back on a square matrix. random weights instead
    R = np.random.default_rng(seed).standard_normal(f([Tensor(x) for x in inputs]).shape)
    loss = lambda ts: (f(ts) * R).sum()

    ts = [Tensor(x.copy()) for x in inputs]
    loss(ts).backward()

    worst = 0.0
    for i, x in enumerate(inputs):
        num = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            old = x[idx]
            x[idx] = old + h
            fp = loss([Tensor(a) for a in inputs]).data
            x[idx] = old - h
            fm = loss([Tensor(a) for a in inputs]).data
            x[idx] = old # put it back
            num[idx] = (fp - fm) / (2 * h)
        worst = max(worst, rel_error(ts[i].grad, num))
    return worst
