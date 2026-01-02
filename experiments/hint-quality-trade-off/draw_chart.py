import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

LABEL_MAP = {
    "-Vanilla-answer-agnostic": "Answer Agnostic",
    "-Vanilla-answer-aware": "Answer Aware"
}

def categorize_convergence(v):
    if v <= 0.2:
        return "Low"
    elif v <= 0.75:
        return "Medium"
    else:
        return "High"

def draw_chart(data, save_path="hint-quality-trade-off.pdf"):
    rows = []
    for model, values in data.items():
        for conv, leak in zip(values["convergence"], values["answer-leakage"]):
            rows.append({
                "Strategy": LABEL_MAP.get(model, model),
                "Convergence": categorize_convergence(conv),
                "Answer Leakage": leak
            })

    df = pd.DataFrame(rows)

    order = ["Low", "Medium", "High"]

    plt.figure(figsize=(6, 4))

    # Violin (transparent background)
    sns.violinplot(
        data=df,
        x="Convergence",
        y="Answer Leakage",
        hue="Strategy",
        order=order,
        inner="box",
        scale="width",
        linewidth=1.2,
        alpha=0.35,      # 🔑 transparency
        zorder=1
    )

    # Points (foreground)
    sns.stripplot(
        data=df,
        x="Convergence",
        y="Answer Leakage",
        hue="Strategy",
        order=order,
        dodge=True,
        jitter=True,
        alpha=0.75,
        size=4,
        zorder=2
    )

    # Fix duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Strategy")

    plt.xlabel("Convergence")
    plt.ylabel("Answer Leakage")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
