#!/usr/bin/env python3
"""Benchmark the ``solver`` package against :mod:`scipy.integrate` (CONTRACT §10).

Compares our ``rk45`` and ``rk4`` integrators (and the compiled Rust path when
``solver.RUST_AVAILABLE``) against ``scipy.integrate.solve_ivp`` on four
problems:

* exponential decay        ``y' = -y``                       (analytic)
* harmonic oscillator      ``y'' = -y``                       (analytic)
* Lotka-Volterra           predator/prey system              (scipy reference)
* a stiff problem (Robertson kinetics) using our ``bdf`` vs scipy ``BDF``.

For every (problem, method, impl) cell we record:

* ``runtime_s``  wall time, ``time.perf_counter``, min of ``N_REPEATS`` runs.
* ``max_error``  max absolute deviation from the analytic solution, or from a
  high-accuracy scipy reference where no closed form exists.
* ``nfev``       number of right-hand-side evaluations reported by the solver.

Everything is deterministic: fixed problem parameters, fixed tolerances, no RNG.

Run as::

    python benchmarks/compare_scipy.py [--out DIR]

Writes ``results.csv`` into ``DIR`` (default: this script's directory) and, when
matplotlib imports, ``runtime.png`` and ``error.png`` alongside it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from math import inf
from time import perf_counter
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp

# --------------------------------------------------------------------------- #
# Make ``solver`` importable even when the package has not been pip-installed,
# by adding the in-repo source root (``python_solver/``) to sys.path.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC_ROOT = os.path.join(_REPO_ROOT, "python_solver")
if os.path.isdir(_SRC_ROOT) and _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import solver  # noqa: E402  (import after sys.path tweak)

# Shared tolerances for the adaptive comparisons (ours vs scipy use the same).
RTOL = 1e-6
ATOL = 1e-9
# Number of timing repeats; the reported runtime is the minimum (CONTRACT: N>=5).
N_REPEATS = 7


# --------------------------------------------------------------------------- #
# Problem definitions
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    name: str
    rhs: Callable[[float, np.ndarray], np.ndarray]
    y0: np.ndarray
    t_span: tuple[float, float]
    # Closed-form solution as a function t-array -> (n, m) array, or None.
    analytic: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # scipy method used to build the high-accuracy reference when analytic is None.
    ref_method: str = "DOP853"
    stiff: bool = False


def _exp_decay() -> Problem:
    def rhs(t, y):
        return -y

    return Problem(
        name="exp_decay",
        rhs=rhs,
        y0=np.array([1.0]),
        t_span=(0.0, 10.0),
        analytic=lambda t: np.array([np.exp(-np.asarray(t, dtype=float))]),
    )


def _harmonic() -> Problem:
    def rhs(t, y):
        # [y, v]' = [v, -y]
        return np.array([y[1], -y[0]])

    def analytic(t):
        t = np.asarray(t, dtype=float)
        return np.array([np.cos(t), -np.sin(t)])

    return Problem(
        name="harmonic",
        rhs=rhs,
        y0=np.array([1.0, 0.0]),
        t_span=(0.0, 20.0),
        analytic=analytic,
    )


def _lotka_volterra() -> Problem:
    a, b, c, d = 1.5, 1.0, 3.0, 1.0

    def rhs(t, y):
        x, z = y[0], y[1]
        return np.array([a * x - b * x * z, -c * z + d * x * z])

    return Problem(
        name="lotka_volterra",
        rhs=rhs,
        y0=np.array([10.0, 5.0]),
        t_span=(0.0, 15.0),
        analytic=None,
        ref_method="DOP853",
    )


def _robertson() -> Problem:
    """Robertson stiff chemical kinetics (classic BDF benchmark)."""

    def rhs(t, y):
        y1, y2, y3 = y[0], y[1], y[2]
        return np.array(
            [
                -0.04 * y1 + 1.0e4 * y2 * y3,
                0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
                3.0e7 * y2 * y2,
            ]
        )

    return Problem(
        name="robertson_stiff",
        rhs=rhs,
        y0=np.array([1.0, 0.0, 0.0]),
        t_span=(0.0, 40.0),
        analytic=None,
        ref_method="Radau",
        stiff=True,
    )


# --------------------------------------------------------------------------- #
# Reference / error helpers
# --------------------------------------------------------------------------- #
def make_reference(problem: Problem) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``ref(t_array) -> (n, m)`` giving the "truth" for error metrics.

    Uses the analytic solution when available, otherwise a tight-tolerance
    scipy dense-output reference.
    """
    if problem.analytic is not None:
        return problem.analytic

    sol = scipy_solve_ivp(
        problem.rhs,
        problem.t_span,
        problem.y0,
        method=problem.ref_method,
        rtol=1e-11,
        atol=1e-12,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(
            f"reference solve failed for {problem.name}: {sol.message}"
        )

    def ref(t_array: np.ndarray) -> np.ndarray:
        return sol.sol(np.asarray(t_array, dtype=float))

    return ref


def max_error(
    t_arr: np.ndarray, y_arr: np.ndarray, ref: Callable[[np.ndarray], np.ndarray]
) -> float:
    """Max absolute deviation of a solution ``(t_arr, y_arr)`` from ``ref``."""
    truth = np.asarray(ref(t_arr), dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    if truth.shape != y_arr.shape:
        # Defensive: align on the overlapping shape.
        m = min(truth.shape[-1], y_arr.shape[-1])
        truth = truth[..., :m]
        y_arr = y_arr[..., :m]
    return float(np.max(np.abs(y_arr - truth)))


# --------------------------------------------------------------------------- #
# Timed runners. Each returns (runtime_s, t_arr, y_arr, nfev).
# --------------------------------------------------------------------------- #
def _time_min(run: Callable[[], object]) -> tuple[float, object]:
    """Call ``run`` ``N_REPEATS`` times, return (min runtime, last result)."""
    best = inf
    result = None
    for _ in range(N_REPEATS):
        start = perf_counter()
        result = run()
        elapsed = perf_counter() - start
        if elapsed < best:
            best = elapsed
    return best, result


def run_ours(problem: Problem, method: str, *, use_rust: bool = False, h=None):
    def run():
        return solver.solve_ivp(
            problem.rhs,
            problem.t_span,
            problem.y0,
            method=method,
            rtol=RTOL,
            atol=ATOL,
            h=h,
            use_rust=use_rust,
        )

    runtime, res = _time_min(run)
    if not res.success:
        raise RuntimeError(f"ours/{method} failed on {problem.name}: {res.message}")
    return runtime, res.t, res.y, int(res.nfev)


def run_scipy(problem: Problem, scipy_method: str):
    def run():
        return scipy_solve_ivp(
            problem.rhs,
            problem.t_span,
            problem.y0,
            method=scipy_method,
            rtol=RTOL,
            atol=ATOL,
        )

    runtime, res = _time_min(run)
    if not res.success:
        raise RuntimeError(
            f"scipy/{scipy_method} failed on {problem.name}: {res.message}"
        )
    return runtime, res.t, res.y, int(res.nfev)


# --------------------------------------------------------------------------- #
# Benchmark orchestration
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    problem: str
    method: str
    impl: str
    runtime_s: float
    max_error: float
    nfev: int


def benchmark() -> list[Row]:
    rows: list[Row] = []

    nonstiff = [_exp_decay(), _harmonic(), _lotka_volterra()]
    stiff = [_robertson()]

    rust = bool(getattr(solver, "RUST_AVAILABLE", False))

    for problem in nonstiff:
        ref = make_reference(problem)

        # Our adaptive Dormand-Prince 5(4).
        rt, t, y, nfev = run_ours(problem, "rk45")
        rows.append(Row(problem.name, "rk45", "ours", rt, max_error(t, y, ref), nfev))

        # Our fixed-step classical RK4.
        rt, t, y, nfev = run_ours(problem, "rk4")
        rows.append(Row(problem.name, "rk4", "ours", rt, max_error(t, y, ref), nfev))

        # scipy's RK45 (also Dormand-Prince 5(4)) for a like-for-like compare.
        rt, t, y, nfev = run_scipy(problem, "RK45")
        rows.append(Row(problem.name, "rk45", "scipy", rt, max_error(t, y, ref), nfev))

        # Compiled Rust path (rk45 + rk4) when the extension is present.
        if rust:
            rt, t, y, nfev = run_ours(problem, "rk45", use_rust=True)
            rows.append(
                Row(problem.name, "rk45", "rust", rt, max_error(t, y, ref), nfev)
            )
            rt, t, y, nfev = run_ours(problem, "rk4", use_rust=True)
            rows.append(
                Row(problem.name, "rk4", "rust", rt, max_error(t, y, ref), nfev)
            )

    for problem in stiff:
        ref = make_reference(problem)

        # Our implicit BDF1/BDF2 (fixed step). Resolve the interval finely enough
        # for the implicit method to track the stiff transient.
        t0, t1 = problem.t_span
        h = (t1 - t0) / 4000.0
        rt, t, y, nfev = run_ours(problem, "bdf", h=h)
        rows.append(Row(problem.name, "bdf", "ours", rt, max_error(t, y, ref), nfev))

        # scipy's variable-order BDF.
        rt, t, y, nfev = run_scipy(problem, "BDF")
        rows.append(Row(problem.name, "bdf", "scipy", rt, max_error(t, y, ref), nfev))

    return rows


# --------------------------------------------------------------------------- #
# Output: CSV, plots, stdout table
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["problem", "method", "impl", "runtime_s", "max_error", "nfev"]


def write_csv(rows: list[Row], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_FIELDS)
        for r in rows:
            writer.writerow(
                [
                    r.problem,
                    r.method,
                    r.impl,
                    f"{r.runtime_s:.9e}",
                    f"{r.max_error:.9e}",
                    r.nfev,
                ]
            )


def print_table(rows: list[Row]) -> None:
    header = ("problem", "method", "impl", "runtime_s", "max_error", "nfev")
    widths = [16, 7, 6, 13, 13, 8]

    def fmt_row(vals):
        cells = []
        for val, w in zip(vals, widths):
            cells.append(str(val).ljust(w))
        return "  ".join(cells)

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt_row(header))
    print(line)
    for r in rows:
        print(
            fmt_row(
                (
                    r.problem,
                    r.method,
                    r.impl,
                    f"{r.runtime_s:.3e}",
                    f"{r.max_error:.3e}",
                    r.nfev,
                )
            )
        )


def make_plots(rows: list[Row], out_dir: str) -> list[str]:
    """Render runtime / error bar charts. Guarded by the caller's try/except."""
    import matplotlib

    matplotlib.use("Agg")  # headless / deterministic backend
    import matplotlib.pyplot as plt

    written: list[str] = []

    labels = [f"{r.problem}\n{r.method}/{r.impl}" for r in rows]
    x = np.arange(len(rows))

    impls = sorted({r.impl for r in rows})
    palette = {
        "ours": "#1f77b4",
        "scipy": "#ff7f0e",
        "rust": "#2ca02c",
    }
    colors = [palette.get(r.impl, "#7f7f7f") for r in rows]

    # Runtime chart (log scale: spans several orders of magnitude).
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))
    ax.bar(x, [max(r.runtime_s, 1e-12) for r in rows], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("runtime [s] (min of %d runs)" % N_REPEATS)
    ax.set_title("solver vs scipy: wall-clock runtime")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette.get(i, "#7f7f7f")) for i in impls
    ]
    ax.legend(handles, impls, title="impl")
    fig.tight_layout()
    runtime_path = os.path.join(out_dir, "runtime.png")
    fig.savefig(runtime_path, dpi=120)
    plt.close(fig)
    written.append(runtime_path)

    # Error chart (log scale; clamp exact-zero error so it remains plottable).
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))
    ax.bar(x, [max(r.max_error, 1e-16) for r in rows], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("max abs error vs reference")
    ax.set_title("solver vs scipy: accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.legend(handles, impls, title="impl")
    fig.tight_layout()
    error_path = os.path.join(out_dir, "error.png")
    fig.savefig(error_path, dpi=120)
    plt.close(fig)
    written.append(error_path)

    return written


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=_HERE,
        help="output directory for results.csv and plots (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rust = bool(getattr(solver, "RUST_AVAILABLE", False))
    print(f"solver.RUST_AVAILABLE = {rust}")
    print(f"running {N_REPEATS} timing repeats per cell (reporting the minimum)\n")

    rows = benchmark()

    csv_path = os.path.join(out_dir, "results.csv")
    write_csv(rows, csv_path)

    print_table(rows)
    print(f"\nwrote {csv_path}")

    try:
        written = make_plots(rows, out_dir)
        for p in written:
            print(f"wrote {p}")
    except Exception as exc:  # matplotlib missing or backend failure: non-fatal
        print(f"(plots skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
