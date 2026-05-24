from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_strategy_difference_heatmap(strategy_difference: pd.DataFrame, output_path: Path) -> None:
    heatmap_data = strategy_difference.pivot(
        index="threshold",
        columns="rewiring_probability",
        values="strategy_difference",
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        origin="lower",
    )

    ax.set_title("Difference in adoption share: clustered minus high-degree seeding")
    ax.set_xlabel("Watts–Strogatz rewiring probability")
    ax.set_ylabel("Adoption threshold")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in heatmap_data.columns])

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in heatmap_data.index])

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Difference in adoption share")

    for y_index, threshold in enumerate(heatmap_data.index):
        for x_index, rewiring_probability in enumerate(heatmap_data.columns):
            value = heatmap_data.loc[threshold, rewiring_probability]
            ax.text(
                x_index,
                y_index,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_threshold_lineplot(aggregated_results: pd.DataFrame, output_path: Path) -> None:
    plot_data = (
        aggregated_results
        .groupby(["threshold", "strategy"], as_index=False)
        .agg(
            mean_final_adoption_share=("mean_final_adoption_share", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, strategy_data in plot_data.groupby("strategy"):
        strategy_data = strategy_data.sort_values("threshold")

        ax.plot(
            strategy_data["threshold"],
            strategy_data["mean_final_adoption_share"],
            marker="o",
            label=strategy,
        )

    ax.set_title("Adoption share by threshold and seeding strategy")
    ax.set_xlabel("Adoption threshold")
    ax.set_ylabel("Mean adoption share after simulation stops")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Seeding strategy")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_clustering_win_rate(clustering_summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = clustering_summary.copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        plot_data["clustering_level"],
        plot_data["clustered_win_rate"],
    )

    ax.set_title("Clustered-seeding win rate by clustering level")
    ax.set_xlabel("Clustering level")
    ax.set_ylabel("Share of settings where clustered seeding wins")
    ax.set_ylim(0, 1.05)

    for index, row in plot_data.iterrows():
        ax.text(
            index,
            row["clustered_win_rate"],
            f"{row['clustered_win_rate']:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_strategy_difference_by_threshold_and_clustering(classified_results: pd.DataFrame, output_path: Path) -> None:
    plot_data = (
        classified_results
        .groupby(["threshold", "clustering_level"], as_index=False)
        .agg(
            mean_strategy_difference=("strategy_difference", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for clustering_level, level_data in plot_data.groupby("clustering_level"):
        level_data = level_data.sort_values("threshold")

        ax.plot(
            level_data["threshold"],
            level_data["mean_strategy_difference"],
            marker="o",
            label=clustering_level,
        )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title("Strategy difference by threshold and clustering level")
    ax.set_xlabel("Adoption threshold")
    ax.set_ylabel("Difference in adoption share")
    ax.legend(title="Clustering level")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_all_plots(data_dir: Path, plots_dir: Path) -> None:
    aggregated_results = pd.read_csv(data_dir / "aggregated_results.csv")
    classified_results = pd.read_csv(data_dir / "strategy_difference_classified.csv")
    strategy_difference = pd.read_csv(data_dir / "strategy_difference.csv")
    clustering_summary = pd.read_csv(data_dir / "clustering_win_rate_summary.csv")

    plot_strategy_difference_heatmap(
        strategy_difference=strategy_difference,
        output_path=plots_dir / "heatmap_strategy_difference.png",
    )

    plot_threshold_lineplot(
        aggregated_results=aggregated_results,
        output_path=plots_dir / "threshold_lineplot.png",
    )

    plot_clustering_win_rate(
        clustering_summary=clustering_summary,
        output_path=plots_dir / "clustering_win_rate.png",
    )

    plot_strategy_difference_by_threshold_and_clustering(
        classified_results=classified_results,
        output_path=plots_dir / "strategy_difference_by_threshold_and_clustering.png",
    )


if __name__ == "__main__":
    create_all_plots(data_dir=Path("data"), plots_dir=Path("plots"))
    print("Saved plots to plots/")