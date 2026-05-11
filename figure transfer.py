import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plotting_helpers import finalize_panel, smart_xlim_focus
from viz_config import get_colors_for_names, format_smart_ticks


# ============================================================
# IO
# ============================================================
def load_results(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# DATA EXTRACTION
# ============================================================
def extract_metrics(results, decoder="linear"):
    data_tr = defaultdict(list)
    data_tr_status = defaultdict(list)

    seed_results = results.get("seed_results", {})

    for seed_data in seed_results.values():
        res = seed_data["results"].get(decoder, {})

        for name, val in res.items():
            delta_tr = val.get("delta_transfer")

            if delta_tr is None or np.isnan(delta_tr):
                continue

            data_tr[name].append(delta_tr)
            data_tr_status[name].append(val.get("transfer_status", ""))


    return data_tr, data_tr_status


# ============================================================
# PLOTTING CORE
# ============================================================
def draw_points_generic(
    ax,
    surrogates,
    data_tr,
    data_tr_status,
    color_map,
    s_scale=1.0,
    alpha=0.8,
):
    for surrogate in surrogates:
        color = color_map.get(surrogate, "gray")

        x_vals = np.asarray(data_tr[surrogate])
        y_vals = 0.02 * (np.random.rand(len(x_vals)) - 0.5)

        statuses = data_tr_status[surrogate]

        for x, y, tr_stat in zip(x_vals, y_vals, statuses):

            marker = "o"

            if "holds" in tr_stat.lower():
                facecolor = "none"
                edgecolor = color
                linewidth = 1.2
            else:
                facecolor = color
                edgecolor = "none"
                linewidth = 0

            ax.scatter(
                x,
                y,
                marker=marker,
                s=400 * s_scale,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )


# ============================================================
# PANEL A
# ============================================================
def plot_panel_A(
    ax,
    data_tr,
    data_tr_status,
    surrogates,
    color_map,
):
    all_x = np.concatenate([data_tr[s] for s in surrogates])



    draw_points_generic(
        ax,
        surrogates,
        data_tr,
        data_tr_status,
        color_map,
        alpha=0.6,
    )

    format_smart_ticks(ax, axis="x", nbins=5)

    ax.axvline(0, color="black", lw=1, alpha=0.2)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    xmin, xmax = smart_xlim_focus(all_x)
    m = max(abs(xmin), abs(xmax))
    ax.set_xlim(-m, m)

    ax.set_ylim(-0.1, 0.1)

    ax.set_yticks([])

    ax.set_xlabel(r"$\Delta$Transfer")
    ax.set_ylabel("")

    finalize_panel(ax, boxed=True)


# ============================================================
# GROUP PANELS
# ============================================================
def plot_group_panel(
    ax,
    data_tr,
    data_tr_status,
    target_labels,
    color_map,
    x_step=None,
    y_step=None,
    show_labels=False,
):
    ax.axvline(0, color="black", lw=1, alpha=0.2)


    valid_labels = [
        l for l in target_labels if l in data_tr and len(data_tr[l]) > 0
    ]

    if valid_labels:
        draw_points_generic(
            ax,
            valid_labels,
            data_tr,
            data_tr_status,
            color_map,
            alpha=0.9,
        )

        x_vals = np.concatenate([data_tr[l] for l in valid_labels])
        ax.set_xlim(*smart_xlim_focus(x_vals))
        ax.set_ylim(-0.1, 0.1)

    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

    x_vals = np.concatenate([data_tr[l] for l in valid_labels])
    ax.set_xlim(*smart_xlim_focus(x_vals))

    unique_x = np.unique(x_vals)

    # =====================================
    # 1 unique point
    # =====================================
    if len(unique_x) == 1:

        ax.set_xticks([unique_x[0]])

    # =====================================
    # 2–3 unique points
    # =====================================
    elif len(unique_x) <= 3:

        rounded_ticks = []

        for x in unique_x:

            if abs(x) < 0.01:
                tick = np.round(x, 4)
            else:
                tick = np.round(x, 2)

            rounded_ticks.append(tick)

        ax.set_xticks(sorted(np.unique(rounded_ticks)))

    # =====================================
    # many points
    # =====================================
    else:

        if x_step is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
        else:
            format_smart_ticks(ax, axis="x")

    def smart_formatter(x, pos):

        if abs(x) < 1e-12:
            return "0"

        abs_x = abs(x)

        if abs_x < 0.01:
            val = f"{x:.5f}".rstrip('0').rstrip('.')
            return val if val not in ["", "-0"] else "0"

        val = f"{x:.2f}".rstrip('0').rstrip('.')
        return val if val not in ["", "-0"] else "0"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(smart_formatter))

    ax.set_yticks([])

    if show_labels:
        ax.set_xlabel(r"$\Delta$Transfer")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    finalize_panel(ax)


def add_panel_titles(fig):

    panel_titles = [
        (0.105, 0.905, "A", ""),
        (0.66, 0.905, "B", ""),
        (0.105, 0.505, "C", ""),
        (0.377, 0.505, "D", ""),
        (0.66, 0.505, "E", ""),
    ]

    for x, y, letter, title in panel_titles:
        fig.text(x, y, letter, fontsize=20, fontweight="bold", color="0.18")
        fig.text(
            x + 0.025,
            y - 0.002,
            title,
            ha="left",
            va="bottom",
            fontsize=20,
            color="0.18",
        )


# ============================================================
# FIGURE
# ============================================================
def make_figure(
    data_tr,
    data_tr_status,
    decoder,
    output_path: Path,
    task_tag=None,
):
    all_surrogates = list(data_tr.keys())

    color_map = dict(
        zip(all_surrogates, get_colors_for_names(all_surrogates))
    )

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1, 0.15], hspace=0.6, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0:2])
    plot_panel_A(
        ax_a,
        data_tr,
        data_tr_status,
        all_surrogates,
        color_map,
    )

    ax_b = fig.add_subplot(gs[0, 2])
    plot_group_panel(
        ax_b,
        data_tr,
        data_tr_status,
        ["Mean-only"],
        color_map,
    )

    ax_c = fig.add_subplot(gs[1, 0])
    plot_group_panel(
        ax_c,
        data_tr,
        data_tr_status,
        ["Copula"],
        color_map,
    )

    ax_d = fig.add_subplot(gs[1, 1])
    plot_group_panel(
        ax_d,
        data_tr,
        data_tr_status,
        ["IID-marginal", "Cov-Gaussian", "Latent-FA"],
        color_map,
    )

    ax_e = fig.add_subplot(gs[1, 2])
    plot_group_panel(
        ax_e,
        data_tr,
        data_tr_status,
        ["Decision-Weak", "Decision-Strong"],
        color_map,
    )

    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")

    surrogate_handles = [
        mlines.Line2D([], [], color=color_map[s], marker="s", ls="", markersize=13, label=s)
        for s in all_surrogates
    ]

    legend_surrogate = ax_leg.legend(
        handles=surrogate_handles,
        loc="upper center",
        ncol=len(all_surrogates),
        fontsize=13,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.2,
        bbox_to_anchor=(0.5, 2.2),
    )
    ax_leg.add_artist(legend_surrogate)

    status_handles = [
        mlines.Line2D([], [], color="black", marker="o", ls="", mfc="none",
                      markersize=13, label="Transfer holds"),
        mlines.Line2D([], [], color="black", marker="o", ls="", mfc="black",
                      markersize=13, label="Transfer violations"),
    ]

    ax_leg.legend(
        handles=status_handles,
        loc="lower center",
        ncol=4,
        fontsize=13,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.2,
        bbox_to_anchor=(0.5, -0.2),
    )


    add_panel_titles(fig)

    return fig


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to the simulation result .pkl file",
    )
    parser.add_argument(
        "--decoder", choices=["linear", "quadratic"], default="quadratic"
    )
    parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    results = load_results(args.results)
    data = extract_metrics(results, decoder=args.decoder)

    task_tag = results.get("data", {}).get("task_type", "task")

    output_path = args.output
    if output_path is None:
        output_path = Path(
            f"Figure_Transfer_{task_tag}_{args.decoder}.pdf"
        )

    fig = make_figure(
        *data,
        decoder=args.decoder,
        output_path=output_path,
        task_tag=task_tag,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    main()
