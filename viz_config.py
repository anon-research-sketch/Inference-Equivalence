import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "figure.dpi": 140,

    "axes.labelsize": 16,
    "text.color": "0.0",
    "axes.labelcolor": "0.0",
    "axes.edgecolor": "0.0",
    "xtick.color": "0.0",
    "ytick.color": "0.0",

    "axes.linewidth": 0.9,

    "xtick.top": False,
    "ytick.right": False,
})

def style_ticks(ax, labelcolor="0.0"):
    ax.tick_params(
        axis="both",
        which="both",
        length=3,
        width=0.8,
        color="0.0",
        labelcolor=labelcolor,
        labelsize=16
    )


def spines_minimal(ax, alpha=0.5, lw=1.5, boxed=False):
    if boxed:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(lw)
            spine.set_alpha(alpha)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

        ax.spines["left"].set_linewidth(lw)
        ax.spines["bottom"].set_linewidth(lw)

        ax.spines["left"].set_alpha(alpha)
        ax.spines["bottom"].set_alpha(alpha)


def crosshair(ax, lw=0.55, alpha=0.035, enabled=True):
    if enabled:
        ax.axhline(0, color="k", lw=lw, alpha=alpha, zorder=0)
        ax.axvline(0, color="k", lw=lw, alpha=alpha, zorder=0)




def annotate_bars(ax, bars, name=None, fmt="{:.2f}", offset=0.05, **kwargs):
    labels = [tick.get_text() for tick in ax.get_xticklabels()]

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if abs(height) > 1e-5:
            current_label = labels[i] if i < len(labels) else ""
            current_offset = 0.15 if current_label == name else offset


            y_pos = height + current_offset
            va = 'bottom'
            default_color = "0.0"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                fmt.format(height),
                fontsize=kwargs.get('fontsize', 12),
                color=kwargs.get('color', default_color),
                ha='center',
                va=va,
                **kwargs
            )


import matplotlib.ticker as ticker


def format_smart_ticks(ax, axis="x", nbins=1):



    target_axis = ax.xaxis if axis == "x" else ax.yaxis

    target_axis.set_major_locator(ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]))



COLORS = {
    "Mean-only":        "#707070",
    "IID-marginal": "tab:green",
    "Cov-Gaussian": "#d62728",
    "Latent-FA": "#800080",
    "Copula": "#BCBD22",
    "Decision-Weak":    "#4C78A8",
    "Decision-Strong":  "#D55E00",
    "identifiable": "#2E5A88",
    "degenerate": "#8B0000",
    "saturated": "#6A5ACD",
}

def get_colors_for_names(names):
    return [COLORS.get(name, "black") for name in names]


