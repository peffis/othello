"""Math sanity checks for the TD(λ) update that don't require TensorFlow.

We verify that the running eligibility update e ← λe + g produces the
textbook closed-form trace e_t = Σ_{k=1..t} λ^(t-k) g_k, and that it
is distinct from what the pre-fix code computed (missing the current
gradient and scaled by an extra λ)."""

import numpy as np

LAMBDA = 0.7


def running_trace(grads):
    e = None
    for g in grads:
        e = g.copy() if e is None else LAMBDA * e + g
    return e


def textbook_trace(grads):
    t = len(grads)
    out = np.zeros_like(grads[0])
    for k, g in enumerate(grads, start=1):
        out = out + (LAMBDA ** (t - k)) * g
    return out


def old_buggy_trace(grads):
    """What the pre-fix code produced at "time t" with t = len(grads).
    It iterated k in range(1, t), i.e. skipped the last gradient, and
    weighted the rest with λ^(t-k) (which at k=t-1 gives λ, not 1).
    """
    t = len(grads)
    out = np.zeros_like(grads[0])
    for k in range(1, t):
        out = out + (LAMBDA ** (t - k)) * grads[k - 1]
    return out


def test_running_matches_textbook():
    grads = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    assert np.allclose(running_trace(grads), textbook_trace(grads))


def test_running_differs_from_old_buggy_form():
    grads = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    assert not np.allclose(running_trace(grads), old_buggy_trace(grads))


def test_single_step():
    grads = [np.array([1.0, 2.0])]
    # After one step, the trace is just the single gradient (weight 1.0).
    assert np.allclose(running_trace(grads), grads[0])
