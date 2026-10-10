"""
how small should eps be in a gradient check, and does float32 vs float64 matter?
python run.py -> prints the tables and writes figures/gradcheck_eps.png (~5s on cpu)

this tests the finite difference side of the check, so it's plain numpy:
the reference grads are either exact (1d function) or a hand written backward (mlp)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DTYPES = {'float64': np.float64, 'float32': np.float32}
OUT = Path(__file__).parent / 'figures' / 'gradcheck_eps.png'

# -----------------------------------------------------------------------------
# 1d function where we know the exact derivative

def f(x):
    return np.sin(x) * np.exp(x)

def df(x):
    return np.exp(x) * (np.sin(x) + np.cos(x))

def fd_errors(eps_list, dtype, central=True, x0=1.3):
    exact = df(x0)
    x = dtype(x0)
    errs = []
    for e in eps_list:
        e = dtype(e) # everything (the shift, f, the subtraction) happens in dtype
        if central:
            g = (f(x + e) - f(x - e)) / (2 * e)
        else:
            g = (f(x + e) - f(x)) / e
        errs.append(abs(float(g) - exact) / abs(exact))
    return np.maximum(errs, 1e-17) # clamp exact zeros so the log plot doesn't choke

# -----------------------------------------------------------------------------
# tiny mlp, 4 -> 8 -> 3 with tanh and cross entropy, 67 params

def make_mlp(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((16, 4))
    y = rng.integers(0, 3, 16)
    P = {
        'W1': rng.standard_normal((4, 8)) * 0.5,
        'b1': rng.standard_normal(8) * 0.1,
        'W2': rng.standard_normal((8, 3)) * 0.5,
        'b2': rng.standard_normal(3) * 0.1,
    }
    return P, X, y

def mlp_loss(P, X, y):
    # stays in whatever dtype P and X are
    h = np.tanh(X @ P['W1'] + P['b1'])
    logits = h @ P['W2'] + P['b2']
    logits = logits - logits.max(1, keepdims=True)
    logprobs = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return -logprobs[np.arange(len(y)), y].mean()

def mlp_backward(P, X, y, bug=0.0):
    # manual backprop in float64, this is the reference. bug scales tanh' by (1 + bug)
    h = np.tanh(X @ P['W1'] + P['b1'])
    logits = h @ P['W2'] + P['b2']
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    dlogits = probs.copy()
    dlogits[np.arange(len(y)), y] -= 1
    dlogits /= len(y)
    dh = dlogits @ P['W2'].T
    dpre = dh * (1 - h**2) * (1 + bug)
    return {'W1': X.T @ dpre, 'b1': dpre.sum(0), 'W2': h.T @ dlogits, 'b2': dlogits.sum(0)}

def mlp_numerical(P, X, y, dtype, eps):
    P = {k: v.astype(dtype) for k, v in P.items()}
    X = X.astype(dtype)
    eps = dtype(eps)
    grads = {}
    for k, v in P.items():
        g = np.zeros(v.shape)
        for idx in np.ndindex(v.shape):
            old = v[idx]
            v[idx] = old + eps
            fp = mlp_loss(P, X, y)
            v[idx] = old - eps
            fm = mlp_loss(P, X, y)
            v[idx] = old
            g[idx] = float(fp - fm) / (2 * float(eps)) # fp - fm is still done in dtype, that's the point
        grads[k] = g
    return grads

def rel_error(A, B):
    a = np.concatenate([A[k].ravel() for k in sorted(A)])
    b = np.concatenate([B[k].ravel() for k in sorted(A)])
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))

# -----------------------------------------------------------------------------
# plotting

BG, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLORS = {'float64': '#2a78d6', 'float32': '#eb6834'}

def mark_min(ax, xs, ys, color, label):
    i = np.argmin(ys)
    ax.scatter(xs[i], ys[i], s=60, color=color, edgecolor=BG, linewidth=2, zorder=3)
    ax.annotate(label.format(eps=xs[i], err=ys[i]), (xs[i], ys[i]), xytext=(0, -16),
                textcoords='offset points', ha='center', va='top', color=INK2, fontsize=9)

def plot(eps_1d, errs_1d, eps_mlp, errs_mlp):
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'text.color': INK, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2, 'font.size': 10, 'legend.frameon': False,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    for name in DTYPES:
        ax1.loglog(eps_1d, errs_1d[name, 'central'], color=COLORS[name], label=f'{name}, central')
        ax1.loglog(eps_1d, errs_1d[name, 'forward'], color=COLORS[name], ls='--', label=f'{name}, forward')
        mark_min(ax1, eps_1d, errs_1d[name, 'central'], COLORS[name], 'best ε ≈ {eps:.0e}')
    ax1.text(1.5e-12, 3e-15, '← rounding error wins', color=MUTED, fontsize=9)
    ax1.text(3e-2, 3e-15, 'truncation error wins →', color=MUTED, fontsize=9, ha='right')
    ax1.set_title('(a) f(x) = sin(x)·eˣ at x = 1.3', loc='left')
    ax1.set_xlabel('step size ε')
    ax1.set_ylabel('relative error of the gradient')
    ax1.set_ylim(1e-15, 3)
    ax1.legend(loc='upper right', fontsize=9)

    for name in DTYPES:
        ax2.loglog(eps_mlp, errs_mlp[name], color=COLORS[name], label=f'{name}, central')
        mark_min(ax2, eps_mlp, errs_mlp[name], COLORS[name], 'best ε ≈ {eps:.0e}\nerror ≈ {err:.0e}')
    ax2.set_title('(b) tiny MLP, all 67 parameters', loc='left')
    ax2.set_xlabel('step size ε')
    ax2.legend(loc='upper right', fontsize=9)

    fig.suptitle('Smaller ε is not better: gradient-check error is U-shaped', x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # rough theory: forward diff is best around sqrt(u), central around u^(1/3)
    print("predicted best eps (u = machine epsilon)")
    for name, dt in DTYPES.items():
        u = np.finfo(dt).eps
        print(f"  {name}: u={u:.1e}  forward {np.sqrt(u):.1e}  central {u**(1/3):.1e}")

    eps_1d = np.logspace(-12, -1, 67)
    errs_1d = {}
    for name, dt in DTYPES.items():
        errs_1d[name, 'central'] = fd_errors(eps_1d, dt, central=True)
        errs_1d[name, 'forward'] = fd_errors(eps_1d, dt, central=False)

    print("\n1d function, measured")
    for (name, kind), errs in errs_1d.items():
        i = np.argmin(errs)
        print(f"  {name} {kind:7s} best eps {eps_1d[i]:.1e}  err {errs[i]:.1e}")

    P, X, y = make_mlp()
    g_ref = mlp_backward(P, X, y)
    # make sure the manual backward is right before trusting it as the reference
    print(f"\nmlp: manual backward vs float64 check at 1e-6: {rel_error(g_ref, mlp_numerical(P, X, y, np.float64, 1e-6)):.1e}")

    eps_mlp = np.logspace(-10, -1, 28)
    errs_mlp = {name: np.array([rel_error(g_ref, mlp_numerical(P, X, y, dt, e)) for e in eps_mlp])
                for name, dt in DTYPES.items()}
    best_eps, floor = {}, {}
    for name, errs in errs_mlp.items():
        i = np.argmin(errs)
        best_eps[name], floor[name] = eps_mlp[i], errs[i]
        at_1e6 = np.interp(-6, np.log10(eps_mlp), errs)
        print(f"  {name} best eps {best_eps[name]:.1e}  err {floor[name]:.1e}  (at eps=1e-6: {at_1e6:.1e})")

    # inject a bug of size delta into the backward and see which precision still notices.
    # "caught" = error at least 10x above that precision's own floor
    print("\nbug size   float64           float32")
    num = {name: mlp_numerical(P, X, y, dt, best_eps[name]) for name, dt in DTYPES.items()}
    for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        g_bug = mlp_backward(P, X, y, bug=delta)
        row = f"  {delta:.0e}  "
        for name in DTYPES:
            err = rel_error(g_bug, num[name])
            row += f" {err:.1e} {'caught' if err > 10 * floor[name] else 'missed'}  "
        print(row)

    plot(eps_1d, errs_1d, eps_mlp, errs_mlp)
    print(f"\nsaved {OUT}")
