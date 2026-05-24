from pathlib import Path

from src.analysis import save_analysis_outputs
from src.experiment import run_experiment_grid
from src.visualization import create_all_plots


def main() -> None:
    raw_results_path = Path("data/raw_results_final.csv")
    data_dir = Path("data")
    plots_dir = Path("plots/final")

    run_experiment_grid(
        n=500,
        k=8,
        thresholds=[
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ],
        rewiring_probabilities=[
            0.00,
            0.01,
            0.05,
            0.10,
            0.20,
            0.40,
            0.80,
            1.00,
        ],
        seed_size=10,
        max_steps=500,
        runs_per_condition=50,
        output_path=raw_results_path,
        base_random_seed=42,
    )

    save_analysis_outputs(
        raw_results_path=raw_results_path,
        output_dir=data_dir,
    )

    create_all_plots(
        data_dir=data_dir,
        plots_dir=plots_dir,
    )


if __name__ == "__main__":
    main()