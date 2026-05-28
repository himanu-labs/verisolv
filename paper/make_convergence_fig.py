"""Generate the convergence figure for the paper: global error vs step size h
on log-log axes for Euler (order 1) and RK4 (order 4) on y' = -y, y(0)=1.

Empirically demonstrates the O(h) rate that EulerConvergence.lean proves, and the
O(h^4) rate of RK4, by overlaying reference slope guides. Output: vector PDF.
Deterministic, no RNG. Run from repo root with the venv active.
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python_solver"))
from solver import solve_ivp  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def endpoint_error(method: str, h: float) -> float:
    res = solve_ivp(lambda t, y: -y, (0.0, 1.0), 1.0, method=method, h=h)
    return abs(res.y_final[0] - np.exp(-1.0))


def main(out: str) -> None:
    hs = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
    euler_err = np.array([endpoint_error("euler", h) for h in hs])
    rk4_err = np.array([endpoint_error("rk4", h) for h in hs])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(hs, euler_err, "o-", color="#c0504d", label="Euler (measured)")
    ax.loglog(hs, rk4_err, "s-", color="#1f497d", label="RK4 (measured)")

    # Reference slope guides anchored at the coarsest step.
    ax.loglog(hs, euler_err[0] * (hs / hs[0]) ** 1, "--", color="#c0504d",
              alpha=0.5, label=r"slope 1 ($O(h)$)")
    ax.loglog(hs, rk4_err[0] * (hs / hs[0]) ** 4, "--", color="#1f497d",
              alpha=0.5, label=r"slope 4 ($O(h^4)$)")

    ax.set_xlabel("step size $h$")
    ax.set_ylabel(r"global error $|y_N - e^{-1}|$")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    # Report fitted slopes (sanity).
    for name, err in (("euler", euler_err), ("rk4", rk4_err)):
        p = np.polyfit(np.log(hs), np.log(err), 1)[0]
        print(f"  fitted order {name}: {p:.3f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "paper/convergence.pdf"
    main(out)
