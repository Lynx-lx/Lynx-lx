"""Expected Loss Criterion (ELC) threshold iteration for risk scores."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_LOSS_WEIGHTS = (1.0, 2.0, 4.0, 8.0)


@dataclass
class ElcResult:
    thresholds: np.ndarray
    elc: float
    history: list[float] = field(default_factory=list)
    n_iter: int = 0
    bin_means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bin_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    converged: bool = False


def _assign_bins(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(thresholds, scores, side="right")


def expected_loss(
    scores: np.ndarray,
    thresholds: np.ndarray,
    loss_weights: np.ndarray,
) -> float:
    """
    ELC = Σ_k λ_k * P(k) * (μ_k + σ_k)

    Intensity μ_k is the mean score in grade k; σ_k is the bin std (scatter).
    Empty bins contribute 0.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    lam = np.asarray(loss_weights, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    bins = _assign_bins(scores, thresholds)
    n = scores.size
    total = 0.0
    for k, w in enumerate(lam):
        xk = scores[bins == k]
        if xk.size == 0:
            continue
        total += float(w * (xk.size / n) * (xk.mean() + xk.std()))
    return total


def iterate_elc_thresholds(
    scores: np.ndarray,
    loss_weights: tuple[float, ...] | np.ndarray = DEFAULT_LOSS_WEIGHTS,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> ElcResult:
    """
    Iterate grade boundaries under the expected-loss criterion.

    Each boundary is the loss-weighted midpoint of adjacent bin means:
        t_k = (λ_k μ_k + λ_{k+1} μ_{k+1}) / (λ_k + λ_{k+1})
    Thresholds are initialized at empirical quantiles and kept monotone.
    """
    x = np.asarray(scores, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    lam = np.asarray(loss_weights, dtype=np.float64)
    if lam.ndim != 1 or lam.size < 2:
        raise ValueError("loss_weights must be a 1D vector with K>=2")
    k_grades = int(lam.size)
    n_bounds = k_grades - 1
    qs = np.linspace(0.0, 1.0, k_grades + 1)[1:-1]

    if x.size == 0:
        t0 = qs.copy()
        return ElcResult(
            thresholds=t0,
            elc=0.0,
            history=[0.0],
            n_iter=0,
            bin_means=np.zeros(k_grades),
            bin_counts=np.zeros(k_grades, dtype=int),
            converged=True,
        )

    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        t = np.linspace(lo, lo + 1e-6, n_bounds + 2)[1:-1]
        elc = expected_loss(x, t, lam)
        return ElcResult(
            thresholds=t,
            elc=elc,
            history=[elc],
            n_iter=0,
            bin_means=np.full(k_grades, lo),
            bin_counts=_bin_counts(x, t, k_grades),
            converged=True,
        )

    t = np.quantile(x, qs)
    history: list[float] = []
    means = np.zeros(k_grades)
    counts = np.zeros(k_grades, dtype=int)
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        bins = _assign_bins(x, t)
        means = np.zeros(k_grades)
        for k in range(k_grades):
            xk = x[bins == k]
            counts[k] = int(xk.size)
            if xk.size:
                means[k] = float(xk.mean())
            else:
                means[k] = float(np.quantile(x, (k + 0.5) / k_grades))
        t_new = (lam[:-1] * means[:-1] + lam[1:] * means[1:]) / (lam[:-1] + lam[1:])
        t_new = np.sort(np.clip(t_new, lo, hi))
        elc = expected_loss(x, t_new, lam)
        history.append(elc)
        if float(np.max(np.abs(t_new - t))) < tol:
            t = t_new
            converged = True
            break
        t = t_new

    return ElcResult(
        thresholds=np.asarray(t, dtype=np.float64),
        elc=float(history[-1] if history else expected_loss(x, t, lam)),
        history=history,
        n_iter=n_iter,
        bin_means=means,
        bin_counts=counts,
        converged=converged,
    )


def _bin_counts(scores: np.ndarray, thresholds: np.ndarray, k_grades: int) -> np.ndarray:
    bins = _assign_bins(scores, thresholds)
    return np.array([(bins == k).sum() for k in range(k_grades)], dtype=int)
