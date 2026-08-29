import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _to_df(matrix_lol):
    labels = [row[0].replace("_", " ").capitalize() for row in matrix_lol]
    values = [row[1:] for row in matrix_lol]
    return pd.DataFrame(values, index=labels, columns=labels)

def draw_chart(all_data, output_path="correlation_heatmaps_row.pdf"):
    panels = [
        ("triviahg", "Spearmans", "TriviaHG", "Spearman’s ρ"),
        ("triviahg", "Kendalltau", "TriviaHG", "Kendall’s τ"),
        ("wikihint", "Spearmans", "WikiHint", "Spearman’s ρ"),
        ("wikihint", "Kendalltau", "WikiHint", "Kendall’s τ"),
    ]

    dfs = [_to_df(all_data[d][m]) for d, m, _, _ in panels]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])

    for i, (ax, df, (_, _, dataset_label, metric_label)) in enumerate(
        zip(axes, dfs, panels)
    ):
        show_cbar = (i == 3)

        sns.heatmap(
            df,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=-0.3,
            vmax=0.3,
            square=True,
            linewidths=0.8,
            linecolor="white",
            annot_kws={"size": 12},
            cbar=show_cbar,
            cbar_ax=cbar_ax if show_cbar else None
        )

        ax.set_title(
            f"{metric_label} ({dataset_label})",
            fontsize=14,
            pad=12
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=13
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=13
        )

    # Colorbar styling
    cbar = axes[-1].collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Correlation coefficient", fontsize=13)

    plt.subplots_adjust(
        left=0.04,
        right=0.90,
        bottom=0.30,
        top=0.88,
        wspace=0.55
    )

    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
