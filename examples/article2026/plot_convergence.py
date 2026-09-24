"""
Plots the convergence history of the iterative scheme solving Eq. (29) of the
paper, as recorded by `6_dislocations_8.py` in its state file.

For every inserted dislocation the residuals r^i = max_d ||Psi_d^i|| are drawn as
markers, joined by a dashed line of the same colour. The horizontal dotted line
marks the level at which the stopping criterion delta^i < delta_tol fires
(delta^i = alpha r^i, so the criterion is met when r^i < delta_tol/alpha).

By default the dashed line is the geometric model r^i = A q^i fitted to the
markers (least squares on log r), which is what the residuals of the RELAXED
scheme follow: they contract with the ratio q = 1 - alpha. At alpha = 1 the
scheme is the classical Newton-Raphson one, the convergence is super-linear and
there are only 2-4 iterations per dislocation, so a geometric fit through them
claims more than the data supports -- pass `--no-fit` to join the markers with a
plain guide line instead.

Usage:
    PYTHONPATH=<wurtzite repo> python3 plot_convergence.py <state file.pkl> [output.pdf] [--skip=2,3] [--no-fit]

`--skip` drops the listed dislocations from the plot (numbered from 1, as in the
paper), e.g. when one of them is discussed separately in the text.
"""
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

# Categorical palette, slots 1-5 (validated for the light surface). Used only
# for state files written before 6_dislocations_8.py started recording the
# colours it draws the dislocations with; normally the palette comes from the
# state file, so that a dislocation wears the same colour here and in the
# lattice figures.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Distinct marker per series, so the identity does not rest on colour alone.
SERIES_MARKERS = ["o", "s", "^", "D", "v"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#a3a29b"
# The state file records lengths in Angstroms; the paper reports them in SI.
ANGSTROM = 1e-10  # [m]
GRID = "#e6e5e1"


def plot_convergence(state, filename, skip=(), fit=True):
    convergence = state["convergence"]
    alpha = state["alpha"]
    tol = state["tol"]
    # Colour and marker follow the dislocation, not its position in the plot,
    # so that `--skip` cannot repaint the series that survive it.
    colors = state.get("colors") or SERIES_COLORS

    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)

    handles = []
    ratios = []
    series = [(n, c) for n, c in enumerate(convergence, start=1)
              if c and n not in skip]
    for n, history in series:
        residuals = np.asarray([r for r, _ in history]) * ANGSTROM
        iterations = np.arange(len(residuals))
        color = colors[(n - 1) % len(colors)]
        marker = SERIES_MARKERS[(n - 1) % len(SERIES_MARKERS)]

        if fit:
            # Geometric model r^i = A q^i fitted in the logarithmic scale, i.e.
            # log r^i = log A + i log q.
            log_q, log_amplitude = np.polyfit(iterations, np.log(residuals), 1)
            ratios.append(float(np.exp(log_q)))
            fine = np.linspace(0, iterations[-1], 100)
            ax.plot(fine, np.exp(log_amplitude) * np.exp(log_q * fine),
                    color=color, linewidth=1.0, linestyle=(0, (4, 2.5)),
                    zorder=2)
        else:
            # A guide for the eye only -- see the module docstring.
            ax.plot(iterations, residuals, color=color, linewidth=1.0,
                    linestyle=(0, (4, 2.5)), zorder=2)
        ax.plot(iterations, residuals, linestyle="none", marker=marker,
                markersize=5.5, color=color, markeredgecolor=TEXT_SECONDARY,
                markeredgewidth=0.6, zorder=3)
        handles.append(Line2D([], [], linestyle="none", marker=marker,
                              markersize=5.5, color=color,
                              markeredgecolor=TEXT_SECONDARY,
                              markeredgewidth=0.6, label=f"$d_{{{n}}}$"))

    # The stopping criterion is imposed on delta^i = alpha r^i.
    stop_level = tol / alpha * ANGSTROM
    ax.axhline(stop_level, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)),
               zorder=1)
    ax.annotate(r"$\delta^i=\delta_\mathrm{tol}$",
                xy=(0.02, stop_level), xycoords=("axes fraction", "data"),
                va="bottom", ha="left", fontsize=8, color=TEXT_SECONDARY)

    ax.set_yscale("log")
    # The iteration counter is an integer; at alpha = 1 the range is so short
    # that matplotlib would otherwise label half-iterations.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("iteration $i$", fontsize=9, color=TEXT_PRIMARY)
    ax.set_ylabel(r"residual $r^i$", fontsize=9, color=TEXT_PRIMARY)
    ax.tick_params(labelsize=8, colors=TEXT_SECONDARY, length=3, width=0.6)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(MUTED)

    legend = ax.legend(handles=handles, loc="upper right", frameon=False,
                       fontsize=8, labelspacing=0.35, handletextpad=0.5,
                       ncol=2, columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.savefig(filename, bbox_inches="tight")
    print(f"Figure saved to {filename}")
    if fit:
        print("fitted ratios q: "
              + ", ".join(f"{n}: {q:.4f}" for (n, _), q in zip(series, ratios)))
        print(f"mean q = {np.mean(ratios):.4f} (1 - alpha = {1 - alpha:.2f})")
    print("iterations to the stopping criterion: "
          + ", ".join(f"d_{n}: {len(c)}" for n, c in series))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fit = "--no-fit" not in sys.argv[1:]
    skip = set()
    for a in sys.argv[1:]:
        if a.startswith("--skip="):
            skip = {int(v) for v in a.split("=", 1)[1].split(",") if v}
    state_file = args[0]
    filename = args[1] if len(args) > 1 else "convergence.pdf"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    plot_convergence(state, filename, skip=skip, fit=fit)


if __name__ == "__main__":
    main()
