from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import pandas as pd
from src.model import ThresholdNetworkModel
from src.seeding import select_clustered_seeds, select_high_degree_seeds


@dataclass(frozen=True)
class SimulationConfig:
    n: int
    k: int
    rewiring_probability: float
    threshold: float
    seed_size: int
    max_steps: int
    strategy: str
    run_id: int
    repetition: int
    network_seed: int
    seeding_seed: int


def run_single_simulation(config: SimulationConfig, graph: nx.Graph | None = None) -> dict:
    rng = random.Random(config.seeding_seed)

    if graph is None:
        graph = nx.watts_strogatz_graph(
            n=config.n,
            k=config.k,
            p=config.rewiring_probability,
            seed=config.network_seed,
        )

    average_clustering = nx.average_clustering(graph)
    average_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()

    initial_seeds = select_initial_seeds(
        graph=graph,
        seed_size=config.seed_size,
        strategy=config.strategy,
        rng=rng,
    )

    model = ThresholdNetworkModel(
        graph=graph,
        threshold=config.threshold,
        initial_seeds=initial_seeds,
        random_seed=config.seeding_seed,
    )

    model.run_until_stable(max_steps=config.max_steps)

    initial_adoption_share = config.seed_size / config.n
    final_adoption_share = model.active_share

    return {
        "run_id": config.run_id,
        "repetition": config.repetition,
        "network_seed": config.network_seed,
        "seeding_seed": config.seeding_seed,
        "strategy": config.strategy,
        "n": config.n,
        "k": config.k,
        "rewiring_probability": config.rewiring_probability,
        "average_clustering": average_clustering,
        "average_degree": average_degree,
        "threshold": config.threshold,
        "seed_size": config.seed_size,
        "max_steps": config.max_steps,
        "steps_until_stable": model.steps,
        "reached_max_steps": model.steps >= config.max_steps and model.running,
        "final_adoption_count": model.active_count,
        "final_adoption_share": final_adoption_share,
        "cascade_happened": final_adoption_share > initial_adoption_share,
        "initial_seeds": ",".join(str(seed) for seed in initial_seeds),
    }


def select_initial_seeds(graph: nx.Graph, seed_size: int, strategy: str, rng: random.Random) -> list[int]:
    if strategy == "high_degree":
        return select_high_degree_seeds(
            graph=graph,
            seed_size=seed_size,
        )

    if strategy == "clustered":
        return select_clustered_seeds(
            graph=graph,
            seed_size=seed_size,
            rng=rng,
        )

    raise ValueError(
        f"Unknown strategy '{strategy}'. "
        "Expected one of: 'high_degree', 'clustered'."
    )


def run_experiment_grid(
    n: int,
    k: int,
    thresholds: list[float],
    rewiring_probabilities: list[float],
    seed_size: int,
    max_steps: int,
    runs_per_condition: int,
    output_path: Path,
    base_random_seed: int = 42,
) -> pd.DataFrame:

    strategies = ["clustered", "high_degree"]

    results: list[dict] = []
    run_id = 0

    for threshold in thresholds:
        for rewiring_probability in rewiring_probabilities:
            for repetition in range(runs_per_condition):
                network_seed = base_random_seed + repetition

                graph = nx.watts_strogatz_graph(
                    n=n,
                    k=k,
                    p=rewiring_probability,
                    seed=network_seed,
                )

                for strategy_index, strategy in enumerate(strategies):
                    seeding_seed = (
                        base_random_seed
                        + 100_000 * repetition
                        + 1_000 * strategy_index
                    )

                    config = SimulationConfig(
                        n=n,
                        k=k,
                        rewiring_probability=rewiring_probability,
                        threshold=threshold,
                        seed_size=seed_size,
                        max_steps=max_steps,
                        strategy=strategy,
                        run_id=run_id,
                        repetition=repetition,
                        network_seed=network_seed,
                        seeding_seed=seeding_seed,
                    )

                    result = run_single_simulation(
                        config=config,
                        graph=graph,
                    )
                    results.append(result)

                    run_id += 1

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    return results_df


if __name__ == "__main__":
    debug_results = run_experiment_grid(
        n=100,
        k=6,
        thresholds=[0.1, 0.2, 0.3],
        rewiring_probabilities=[0.0, 0.1, 0.5],
        seed_size=5,
        max_steps=100,
        runs_per_condition=3,
        output_path=Path("data/raw_results_debug.csv"),
        base_random_seed=42,
    )

    print(debug_results.head())
    print(f"\nSaved {len(debug_results)} rows to data/raw_results_debug.csv")