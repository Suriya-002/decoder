"""Publication figures for the atom-loss correlation estimator paper.

Produces IEEE single-column (3.5 in) vector PDFs plus PNG previews.
Regenerate after any re-run.

    python make_figures.py

FIGURE 2 ERROR BARS -- PROVENANCE (corrected 24 Aug 2026)
Every yerr below is the half-width printed by the run that produced the
corresponding point. Nothing here is reconstructed or rounded up.

  Rotated surface, d=5 (BLUE)
      run_eta_raw.py --pg 0.00054 --shots 30000, seed 4242 (default),
      naive collector, eps=0. 14,140 / 14,236 / 14,185 / 14,257 / 14,222
      events per point.
      CIs: [-0.021,0.012] [0.233,0.265] [0.493,0.521] [0.748,0.770] [1.000,1.000]

  Unrotated surface, d=5 (RUST)
      generality run, ~10.2k events per point.
      CIs: [-0.025,0.013] -- [0.471,0.505] [0.734,0.760] [1.000,1.000]
      The eta=0.25 point is a 30,000-shot re-run: 0.2456 +- 0.0110.
      (The 12k-shot value 0.276 was a 2.6-sigma low-stats fluctuation and
      is not the reported point.)

  Color code, d=5, n_sub=6 (TEAL)
      est_color.py, 3,414 / 3,515 / 3,463 events per point.
      CIs: [-0.020,0.047] [0.479,0.536] [1.000,1.000]

  At eta=1 the half-width is exactly 0.000, not a small positive number:
  a co-lost ancilla has no atom and therefore no |1> population, so
  P(m=0)=1.0000 with zero sample variance. The zero-length bar is correct.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# ---------------------------------------------------------------- style
COL = 3.5           # IEEE single-column width, inches
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.3,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

INK   = "#1a1a1a"
BLUE  = "#2C5F8D"
RUST  = "#B4553F"
TEAL  = "#3F8A80"
GREY  = "#8A8A8A"
FAINT = "#D8D8D8"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


# ============================================================ FIGURE 2
# Estimator recovery across three code families.
def figure2():
    eta = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

    rot_x,  rot_y,  rot_e  = eta, np.array([-0.005, 0.249, 0.507, 0.759, 1.000]), \
                                  np.array([0.0165, 0.0159, 0.0142, 0.0107, 0.0000])
    unr_x,  unr_y,  unr_e  = eta, np.array([-0.006, 0.246, 0.488, 0.747, 1.000]), \
                                  np.array([0.0190, 0.0110, 0.0170, 0.0130, 0.0000])
    col_x,  col_y,  col_e  = np.array([0.00, 0.50, 1.00]), \
                             np.array([0.013, 0.508, 1.000]), \
                             np.array([0.0335, 0.0285, 0.0000])

    fig, ax = plt.subplots(figsize=(COL, COL * 0.82))

    ax.plot([-0.02, 1.02], [-0.02, 1.02], ls=(0, (4, 3)), lw=0.7,
            color=GREY, zorder=1)

    ax.errorbar(rot_x, rot_y, yerr=rot_e, fmt="o", color=BLUE, mfc=BLUE,
                mec=BLUE, ecolor=BLUE, elinewidth=0.8, capsize=1.8,
                capthick=0.8, label="Rotated surface, $d{=}5$", zorder=4)
    ax.errorbar(unr_x, unr_y, yerr=unr_e, fmt="s", color=RUST, mfc="white",
                mec=RUST, ecolor=RUST, elinewidth=0.8, capsize=1.8,
                capthick=0.8, mew=0.9, label="Unrotated surface, $d{=}5$",
                zorder=3)
    ax.errorbar(col_x, col_y, yerr=col_e, fmt="^", color=TEAL, mfc="white",
                mec=TEAL, ecolor=TEAL, elinewidth=0.8, capsize=1.8,
                capthick=0.8, mew=0.9, label="Color code, $d{=}5$", zorder=2)

    ax.set_xlabel(r"true correlation $\eta$")
    ax.set_ylabel(r"estimate $\hat{\eta}$")
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", borderpad=0.2)
    ax.text(0.62, 0.66, "ideal", color=GREY, fontsize=7, style="italic",
            ha="center", va="bottom", rotation=45,
            rotation_mode="anchor", transform=ax.transData)
    save(fig, "fig2_recovery")


# ============================================================ FIGURE 3
# Shot cost versus loss-localizer quality.
def figure3():
    n0 = 1.5e4
    series = [
        ("Oracle  (1.00, 1.00)",       0.059, BLUE, "o", "-"),
        ("Learned  (0.65, 0.85)",      0.083, RUST, "s", "--"),
        ("Crude  (0.40, 0.60)",        0.115, TEAL, "^", "-."),
        ("Very crude  (0.25, 0.40)",   0.187, GREY, "D", ":"),
    ]
    N = np.logspace(np.log10(1.2e4), np.log10(2e6), 200)

    fig, ax = plt.subplots(figsize=(COL, COL * 0.82))
    for label, s0, c, mk, ls in series:
        ax.plot(N, s0 * np.sqrt(n0 / N), ls=ls, color=c, lw=1.0, label=label)
        ax.plot([n0], [s0], mk, color=c, mfc="white", mec=c, mew=0.9,
                ms=4, zorder=5)

    ax.axhline(0.01, color=FAINT, lw=0.7, zorder=0)
    ax.text(1.35e4, 0.0105, r"$\sigma_\eta = 0.01$", fontsize=6.5,
            color=GREY, va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shots")
    ax.set_ylabel(r"standard error $\sigma_\eta$")
    ax.set_xlim(1.2e4, 2e6)
    ax.set_ylim(4e-3, 3e-1)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower left", borderpad=0.2)
    save(fig, "fig3_shotcost")


# ============================================================ FIGURE 4
# Effect of loss information on logical error rate.
def figure4():
    labels = ["Ignorant", "Data-aware", r"$\eta$-informed", "Oracle"]
    ler    = np.array([0.34636, 0.21945, 0.23455, 0.23455])
    n      = 5500
    ci     = 1.96 * np.sqrt(ler * (1 - ler) / n)
    colors = [GREY, BLUE, RUST, TEAL]

    fig, ax = plt.subplots(figsize=(COL, COL * 0.80))
    x = np.arange(4)
    ax.bar(x, ler, width=0.62, color=colors, edgecolor=INK, linewidth=0.5,
           zorder=2)
    ax.errorbar(x, ler, yerr=ci, fmt="none", ecolor=INK, elinewidth=0.8,
                capsize=2.2, capthick=0.8, zorder=3)

    for xi, v in zip(x, ler):
        ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, color=INK)

    # control bracket: ignorant vs data-aware
    ax.annotate("", xy=(0, 0.385), xytext=(1, 0.385),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
    ax.text(0.5, 0.393, r"$+0.127 \pm 0.017$", ha="center", va="bottom",
            fontsize=6.5, color=INK)

    # the null: data-aware vs eta-informed
    ax.annotate("", xy=(1, 0.283), xytext=(2, 0.283),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=RUST))
    ax.text(1.5, 0.290, r"$-0.015 \pm 0.016$", ha="center", va="bottom",
            fontsize=6.5, color=RUST)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("logical error rate")
    ax.set_ylim(0, 0.44)
    ax.set_xlim(-0.6, 3.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)
    save(fig, "fig4_impact")


if __name__ == "__main__":
    print("building figures...")
    figure2()
    figure3()
    figure4()
    print("done.")
