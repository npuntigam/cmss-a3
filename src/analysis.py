
from __future__ import annotations
from pathlib import Path
import pandas as pd


def aggregate_strategy_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw simulation runs by threshold, rewiring probability & strat
    Output = 1 row per threshold × rewiring_probability × strategy
    """
    grouped = (
        raw_results
        .groupby(
            [
                "threshold",
                "rewiring_probability",
                "strategy",
            ],
            as_index=False,
        )
        .agg(
            mean_final_adoption_share=("final_adoption_share", "mean"),
            std_final_adoption_share=("final_adoption_share", "std"),
            mean_final_adoption_count=("final_adoption_count", "mean"),
            mean_steps_until_stable=("steps_until_stable", "mean"),
            cascade_rate=("cascade_happened", "mean"),
            mean_average_clustering=("average_clustering", "mean"),
            mean_average_degree=("average_degree", "mean"),
            number_of_runs=("run_id", "count"),
        )
    )

    return grouped


def compute_strategy_difference(aggregated_results: pd.DataFrame) -> pd.DataFrame:
    performance_pivot = aggregated_results.pivot_table(
        index=[
            "threshold",
            "rewiring_probability",
        ],
        columns="strategy",
        values="mean_final_adoption_share",
        aggfunc="mean",
    ).reset_index()

    required_columns = {"clustered", "high_degree"}
    missing_columns = required_columns - set(performance_pivot.columns)

    if missing_columns:
        raise ValueError(
            f"Cannot compute strategy difference. Missing strategy columns: "
            f"{sorted(missing_columns)}"
        )

    clustering_summary = (
        aggregated_results
        .groupby(["threshold", "rewiring_probability"], as_index=False)
        .agg(
            mean_average_clustering=("mean_average_clustering", "mean"),
            mean_average_degree=("mean_average_degree", "mean"),
            number_of_runs=("number_of_runs", "sum"),
        )
    )

    result = performance_pivot.merge(
        clustering_summary,
        on=["threshold", "rewiring_probability"],
        how="left",
    )

    result["strategy_difference"] = result["clustered"] - result["high_degree"]
    result["clustered_outperforms"] = result["strategy_difference"] > 0

    return result.rename_axis(columns=None)


def classify_clustering_level(strategy_difference: pd.DataFrame, quantile: float = 0.5) -> pd.DataFrame:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")

    clustering_cutoff = strategy_difference["mean_average_clustering"].quantile(
        quantile
    )

    result = strategy_difference.copy()

    result["clustering_level"] = result["mean_average_clustering"].apply(
        lambda value: (
            "highly_clustered"
            if value > clustering_cutoff
            else "weakly_clustered"
        )
    )

    return result


def summarize_clustered_win_rate_by_clustering(classified_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        classified_results
        .groupby("clustering_level", as_index=False)
        .agg(
            clustered_win_rate=("clustered_outperforms", "mean"),
            number_of_parameter_settings=("clustered_outperforms", "count"),
            mean_strategy_difference=("strategy_difference", "mean"),
            mean_average_clustering=("mean_average_clustering", "mean"),
        )
    )

    return summary


def save_analysis_outputs(raw_results_path: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    raw_results = pd.read_csv(raw_results_path)

    aggregated_results = aggregate_strategy_results(raw_results)
    strategy_difference = compute_strategy_difference(aggregated_results)
    classified_results = classify_clustering_level(strategy_difference)
    clustering_summary = summarize_clustered_win_rate_by_clustering(
        classified_results
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_results.to_csv(
        output_dir / "aggregated_results.csv",
        index=False,
    )
    strategy_difference.to_csv(
        output_dir / "strategy_difference.csv",
        index=False,
    )
    classified_results.to_csv(
        output_dir / "strategy_difference_classified.csv",
        index=False,
    )
    clustering_summary.to_csv(
        output_dir / "clustering_win_rate_summary.csv",
        index=False,
    )

    return {
        "aggregated_results": aggregated_results,
        "strategy_difference": strategy_difference,
        "classified_results": classified_results,
        "clustering_summary": clustering_summary,
    }


if __name__ == "__main__":
    outputs = save_analysis_outputs(
        raw_results_path=Path("data/raw_results_debug.csv"),
        output_dir=Path("data"),
    )

    print("\nStrategy difference:")
    print(outputs["strategy_difference"])

    print("\nClustering summary:")
    print(outputs["clustering_summary"])