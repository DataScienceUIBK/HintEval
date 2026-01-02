import math
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


def draw_chart(
    data: Dict[str, Dict[str, float]],
    out_pdf: str = "vanilla-vs-finetuning.pdf",
    title: str = "Radar Chart",
    figsize: Tuple[float, float] = (4.8, 4.8),
    label_radius: float = 1.10,   # move labels in/out (try 1.06~1.14)
    fill_alpha: float = 0.12,
) -> str:
    """
    data: {"-Vanilla": {"relevance": 0.9, ...}, "-Finetuned": {...}}
    Saves PDF and returns output path.
    """

    if not data:
        raise ValueError("`data` is empty.")

    # metric order = insertion order from first series
    first_key = next(iter(data))
    metrics = list(data[first_key].keys())
    if not metrics:
        raise ValueError("No metrics found in the first series.")

    # Validate consistent metrics
    for name, vals in data.items():
        if set(vals.keys()) != set(metrics):
            raise ValueError(f"Series '{name}' has different metric keys than the first series.")

    def cap_metric(s: str) -> str:
        return s.replace("_", " ").replace("-", " ").strip().title()

    def cap_series(s: str) -> str:
        s = s.strip()
        if s.startswith("-"):
            s = s[1:].strip()
        return s[:1].upper() + s[1:]

    metric_labels = [cap_metric(m) for m in metrics]

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)   # start at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_ylim(0.0, 1.0)

    # y ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

    # x ticks: we draw our own labels, so blank them out
    ax.set_xticks(angles)
    ax.set_xticklabels([""] * n)

    # plot series
    for series_name, series_vals in data.items():
        values = [float(series_vals[m]) for m in metrics]
        values_closed = values + values[:1]
        ax.plot(angles_closed, values_closed, linewidth=2, label=cap_series(series_name))
        ax.fill(angles_closed, values_closed, alpha=fill_alpha)

    # ---- Custom rotated labels, CENTERED on the radial line ----
    for theta, lbl in zip(angles, metric_labels):
        theta_deg = np.degrees(theta)

        # Tangent direction at that point.
        # (With theta_offset and clockwise, this formula gives the "sloped along circle" look.)
        rot = -theta_deg

        # Keep upright-ish (avoid upside down)
        if rot < -90:
            rot += 180
        if rot > 90:
            rot -= 180

        ax.text(
            theta,
            label_radius,
            lbl,
            fontsize=11,
            rotation=rot,
            rotation_mode="anchor",
            ha="center",   # <-- key change: center on the radial line
            va="center",
        )

    # ax.set_title(title.strip().title(), fontsize=18, fontweight="bold", pad=16)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, len(data)),
        frameon=False,
        fontsize=12,
    )

    fig.subplots_adjust(top=0.88, bottom=0.24, left=0.06, right=0.94)

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_pdf
