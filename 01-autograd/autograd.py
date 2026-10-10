"""
tiny autograd engine on top of numpy.
same idea as micrograd, but every node is an array instead of a single number.
"""

import numpy as np


def unbroadcast(grad, shape):
    # numpy broadcasts silently in the forward pass, so in the backward pass
    # we have to sum the grad back down to the shape the input actually had
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:

    __array_ufunc__ = None # makes ndarray + Tensor fall through to Tensor.__radd__

    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        # graph bookkeeping for backward()
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that made this node, only for debugging

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only int/float powers for now"
        out = Tensor(self.data**n, (self,), f'**{n}')
        def _backward():
            self.grad += n * self.data**(n-1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # todo: 1-d @ 2-d isn't handled, haven't needed it yet
        assert self.data.ndim >= 2 and other.data.ndim >= 2
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            # swapaxes instead of .T so batched inputs (B, T, C) work too, attention needs this later
            self.grad += unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.shape)
            other.grad += unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis) # put the reduced axis back so it broadcasts
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        return s * (s.data.size / self.data.size)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), (self,), 'transpose')
        inv = None if axes is None else np.argsort(axes) # inverse permutation
        def _backward():
            self.grad += out.grad.transpose(inv)
        out._backward = _backward
        return out

    def backward(self):
        # topological order of everything in the graph.
        # explicit stack instead of recursion, the recursive version hits python's
        # recursion limit once the graph is ~1000 ops deep
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            v, children_done = stack.pop()
            if children_done:
                topo.append(v) # all of v's inputs are already in topo
                continue
            if v in visited:
                continue
            visited.add(v)
            stack.append((v, True))
            for child in v._prev:
                if child not in visited:
                    stack.append((child, False))

        # walk it in reverse and apply the chain rule one node at a time.
        # note: grads accumulate (+=), so reset them between steps when training
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"


def softmax(x, axis=-1):
    # subtract the max first or exp() overflows on big logits.
    # softmax doesn't change under a constant shift, so the max is just a constant here (no grad through it)
    x = x - x.data.max(axis=axis, keepdims=True)
    e = x.exp()
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(logits, targets):
    # logits (N, C), targets (N,) of class ids -> mean negative log likelihood
    logits = logits - logits.data.max(axis=-1, keepdims=True)
    logprobs = logits - logits.exp().sum(axis=-1, keepdims=True).log()
    onehot = np.eye(logits.shape[-1])[targets]
    return -(logprobs * onehot).sum() / len(targets)
