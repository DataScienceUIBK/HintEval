import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ---------------- Styling ----------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AWARE_COLOR    = "#E45756"
AGNOSTIC_COLOR = "#4C72B0"
DELTA_COLOR    = "#7F7F7F"

# Frame palettes
UP_EDGE   = "#2E7D32"
UP_FILL   = "#E8F5E9"
DOWN_EDGE = "#EF6C00"
DOWN_FILL = "#FFF3E0"


# ---------------- Helper plotting functions ----------------
def _union_bbox(ax_list):
    bboxes = [ax.get_position() for ax in ax_list]
    xmin, ymin = min(b.x0 for b in bboxes), min(b.y0 for b in bboxes)
    xmax, ymax = max(b.x1 for b in bboxes), max(b.y1 for b in bboxes)
    return xmin, ymin, xmax, ymax


def _add_fancy_group_frame(
    fig,
    ax_list,
    title_text,
    edge_color,
    fill_color,
    pad_left=0.06,
    pad_right=0.06,
    pad_bottom=0.08,
    pad_top=0.13,
    rounding=0.02,
    title_y_offset=0.02,
):
    xmin, ymin, xmax, ymax = _union_bbox(ax_list)

    bg = FancyBboxPatch(
        (xmin - pad_left, ymin - pad_bottom),
        (xmax - xmin) + pad_left + pad_right,
        (ymax - ymin) + pad_bottom + pad_top,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=0.0,
        edgecolor="none",
        facecolor=fill_color,
        alpha=0.55,
        transform=fig.transFigure,
        zorder=-5,
    )
    bg.set_path_effects([
        pe.SimplePatchShadow(offset=(1.6, -1.6), alpha=0.25, rho=0.95),
        pe.Normal()
    ])
    fig.patches.append(bg)

    outline = FancyBboxPatch(
        (xmin - pad_left, ymin - pad_bottom),
        (xmax - xmin) + pad_left + pad_right,
        (ymax - ymin) + pad_bottom + pad_top,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=1.6,
        edgecolor=edge_color,
        facecolor="none",
        linestyle=(0, (4, 3)),
        transform=fig.transFigure,
        zorder=20,
    )
    outline.set_path_effects([
        pe.Stroke(linewidth=2.6, foreground="white", alpha=0.75),
        pe.Normal()
    ])
    fig.patches.append(outline)

    pill_w, pill_h = 0.08, 0.08
    cx = (xmin + xmax) / 2
    pill_x = cx - pill_w / 2
    pill_y = (ymax + title_y_offset) + 0.15

    # pill = FancyBboxPatch(
    #     (pill_x, pill_y),
    #     pill_w, pill_h,
    #     boxstyle="round,pad=0.01,rounding_size=0.08",
    #     linewidth=1.2,
    #     edgecolor=edge_color,
    #     facecolor="white",
    #     transform=fig.transFigure,
    #     zorder=30,
    # )
    # pill.set_path_effects([
    #     pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.18, rho=0.98),
    #     pe.Normal()
    # ])

    pill = FancyBboxPatch(
        (pill_x, pill_y),
        pill_w, pill_h,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        linewidth=1.1,
        edgecolor=edge_color,
        facecolor="white",
        transform=fig.transFigure,
        zorder=30,
    )

    # subtle shadow + crisp edge (lighter than before)
    pill.set_path_effects([
        pe.SimplePatchShadow(offset=(0.8, -0.8), alpha=0.12, rho=0.98),
        pe.Normal()
    ])

    fig.patches.append(pill)

    fig.text(
        cx, pill_y + pill_h / 2,
        title_text,
        ha="center", va="center",
        fontsize=10, weight="bold",
        color=edge_color,
        zorder=31
    )


def _draw_up_frame(fig, ax_list):
    _add_fancy_group_frame(
        fig, ax_list,
        title_text="↑ Higher is Better",
        edge_color=UP_EDGE,
        fill_color=UP_FILL,
        pad_left=0,
        pad_right=0,
        pad_bottom=0.14,
        pad_top=0.3,
        rounding=0.02,
        title_y_offset=0.02,
    )


def _draw_down_frame(fig, ax_list):
    _add_fancy_group_frame(
        fig, ax_list,
        title_text="↓ Lower is Better",
        edge_color=DOWN_EDGE,
        fill_color=DOWN_FILL,
        pad_left=0.0,
        pad_right=0.0,
        pad_bottom=0.14,
        pad_top=0.3,
        rounding=0.02,
        title_y_offset=0.02,
    )


def _plot_metric(ax, metric, pivot, gen_order):
    aware_vals = pivot[(metric, "Aware")].values
    agn_vals = pivot[(metric, "Agnostic")].values
    y = np.arange(len(gen_order))

    all_vals = np.concatenate([aware_vals, agn_vals])
    spread = all_vals.max() - all_vals.min()
    pad = spread * 0.25 if spread > 0 else 0.1
    ax.set_xlim(all_vals.min() - pad, all_vals.max() + pad)

    ax.barh(y, agn_vals, height=0.55, color=AGNOSTIC_COLOR, alpha=0.35)
    ax.barh(y, aware_vals, height=0.30, color=AWARE_COLOR, alpha=0.90)

    for yi, a, g in zip(y, aware_vals, agn_vals):
        ax.plot([g, a], [yi, yi], color=DELTA_COLOR, linewidth=1.6)

    ax.set_title(metric)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(axis="y", length=0, pad=15)
    ax.invert_yaxis()


# ---------------- Main API ----------------
def draw_chart(
    data,
    pdf_path="./aware-vs-agnostic.pdf",
    gen_order=("LLaMA-3.1-8b", "LLaMA-3.1-70b", "LLaMA-3.1-405b", "GPT-4"),
    group_up=("Relevance", "Convergence", "Familiarity"),
    group_down=("Readability", "Answer Leakage"),
    figsize=(12, 1.6),
):
    """
    data: list[tuple] of rows like:
      (Generator, Strategy, Relevance, Readability, Convergence, Familiarity, Answer Leakage)

    Saves a PDF to pdf_path.
    """
    cols = ["Generator","Strategy","Relevance","Readability","Convergence","Familiarity","Answer Leakage"]
    df = pd.DataFrame(data, columns=cols)

    metrics = list(group_up) + list(group_down)

    # Ensure the intended generator ordering is respected
    gen_order = list(gen_order)
    pivot = (
        df.pivot(index="Generator", columns="Strategy", values=metrics)
          .reindex(gen_order)
    )

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True)

    for ax, metric in zip(axes, metrics):
        _plot_metric(ax, metric, pivot=pivot, gen_order=gen_order)
        ax.set_facecolor("none")

    axes[0].set_yticks(np.arange(len(gen_order)))
    axes[0].set_yticklabels(gen_order)

    _draw_up_frame(fig, axes[:len(group_up)])
    _draw_down_frame(fig, axes[len(group_up):])

    legend_elements = [
        Line2D([0], [0], color=AGNOSTIC_COLOR, lw=6, alpha=0.4, label="Answer-Agnostic"),
        Line2D([0], [0], color=AWARE_COLOR,   lw=6, alpha=0.9, label="Answer-Aware"),
        Line2D([0], [0], color=DELTA_COLOR,   lw=2, label="Δ (Aware − Agnostic)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),  # ↑ increase this second value to move legend up
        ncol=3,
        frameon=False,
    )
    fig.suptitle(" ", y=1.12, fontsize=13)
    # plt.tight_layout(pad=0.8)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path
