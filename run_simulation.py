import argparse
import pickle

import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from config import (
    EXPERIMENT_SEED,
    N,
    NOISE,
    SEEDS,
    TASK,
    TASK_TYPE,
    WORLD_SEED,
    make_rng,
)
from data import generate_base_world, generate_fixed_dataset
from ie_evaluation import run_ie_both_decoders
from ie_interventions import (
    IE_INTERVENTIONS,
    descriptive_benchmark
)



def run(
    noise=NOISE,
    task=TASK,
    task_type=TASK_TYPE,
    world_seed=WORLD_SEED,
    n=N,
    decoder_types=None,
    decoder_tag=None,
    save_results=True,
    verbose=True,
):
    """
    Run the full inference-equivalence simulation across all seeds.

    Workflow
    --------
    1. Generate a fixed latent world
    2. Simulate datasets across seeds
    3. Compute diagnostic margin statistics
    4. Evaluate IE under decoder families
    5. Save per-seed and aggregated outputs
    """
    folder_name = f"ie_sim_{task_type}"

    tag = (
        decoder_tag
        if decoder_tag
        else (
            "_".join(decoder_types)
            if decoder_types
            else "both"
        )
    )

    RESULTS_DIR = Path("ie_sim")
    task_dir = RESULTS_DIR / folder_name
    seed_dir = task_dir / f"per_seed_{tag}"
    sim_results_path = task_dir / f"ie_results_{tag}.pkl"

    zn, zt = generate_base_world(
        n=n,
        seed=world_seed,
    )

    master = {
        "data": {
            "noise": noise,
            "task": task,
            "task_type": task_type,
            "decoder_types": decoder_types,
            "decoder_tag": tag,
            "experiment_seed": EXPERIMENT_SEED,
        },
        "seed_results": {},
        "diagnostics": {
            "margin_corr": {},
            "margin_scatter": {},
            "descriptive_stats": {},
        },
    }

    if save_results:
        seed_dir.mkdir(parents=True, exist_ok=True)

    all_desc_stats = []

    for seed in tqdm(SEEDS, desc="Processing Seeds"):
        seed_file = seed_dir / f"seed_{seed}.pkl"

        if save_results and seed_file.exists():
            if verbose:
                tqdm.write(
                    f"Skipping seed {seed} "
                    "(already saved)"
                )

            with open(seed_file, "rb") as f:
                loaded = pickle.load(f)

            master["seed_results"][seed] = loaded[
                "result"
            ]



            if "descriptive_stats" in loaded:
                all_desc_stats.append(
                    loaded["descriptive_stats"]
                )


            continue

        x, y = generate_fixed_dataset(
            zn,
            zt,
            seed=seed,
            noise=noise,
            task=task,
            task_type=task_type,
        )


        current_desc_stats = {}





        for (
            name,
            builder,
        ) in IE_INTERVENTIONS.items():

            local_rng = make_rng(seed + hash(name) % 10000)


            desc = descriptive_benchmark(
                x,
                y,
                builder,
                local_rng,
            )

            current_desc_stats[name] = desc


            if verbose:

                metric_order = [
                    "mean_diff",
                    "var_diff",
                    "cov_corr",
                    "cov_frobenius",
                    "ev_diff",
                ]

                stat_parts = []

                for key in metric_order:
                    if key in desc:
                        stat_parts.append(
                            f"{key}={desc[key]:.4f}"
                        )

                tqdm.write(
                    f"[Seed {seed} | {name}] "
                    + ", ".join(stat_parts)
                )




        all_desc_stats.append(current_desc_stats)



        result = run_ie_both_decoders(
            x,
            y,
            seed=seed,
            decoder_types=decoder_types,
            verbose=verbose,
        )

        master["seed_results"][seed] = result

        if save_results:
            data_to_save = {
                "seed": seed,
                "task_type": task_type,
                "decoder_types": decoder_types,
                "result": result,
                "noise": noise,
                "task": task,
                "descriptive_stats": current_desc_stats,
            }



            with open(seed_file, "wb") as f:
                pickle.dump(data_to_save, f)



    if all_desc_stats:
        master["diagnostics"]["descriptive_stats"] = (
            aggregate_descriptive_stats(
                all_desc_stats
            )
        )

    if save_results:
        sim_results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbose:
            tqdm.write(
                f"\nSaving data to "
                f"{sim_results_path}"
            )

        with open(
            sim_results_path,
            "wb",
        ) as f:
            pickle.dump(master, f)

    return master

def aggregate_descriptive_stats(all_desc_stats):

    summary = {}

    surrogate_names = all_desc_stats[0].keys()

    for name in surrogate_names:

        metric_names = all_desc_stats[0][name].keys()

        summary[name] = {}

        for metric in metric_names:

            vals = [
                seed_stats[name][metric]
                for seed_stats in all_desc_stats
            ]

            summary[name][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    return summary

def print_summary(master, verbose=True):
    """
    Print simulation summary statistics
    in paper-ready format.
    """
    if not verbose:
        return

    seed_results = master["seed_results"]
    n_seeds = len(seed_results)

    if n_seeds == 0:
        print("No results to summarize")
        return


    regime_counts = {
        "linear": {
            "identifiable": 0,
            "saturated": 0,
            "degenerate": 0,
        },
        "quadratic": {
            "identifiable": 0,
            "saturated": 0,
            "degenerate": 0,
        },
    }

    for result in seed_results.values():
        for (
            decoder,
            decoder_results,
        ) in result["results"].items():
            for vals in decoder_results.values():
                regime_counts[decoder][
                    vals["regime"]
                ] += 1

            for (
                surrogate_name,
                vals,
            ) in decoder_results.items():
                if (
                    vals["regime"]
                    != "identifiable"
                ):
                    continue




    print("\nREGIME DISTRIBUTION:")

    for decoder in ["linear", "quadratic"]:
        total = sum(
            regime_counts[decoder].values()
        )

        if total == 0:
            continue

        print(f"{decoder.upper():<10} | ", end="")

        for regime in [
            "identifiable",
            "saturated",
            "degenerate",
        ]:
            pct = (
                regime_counts[decoder][regime]
                / total
            )
            print(
                f"{regime}:{pct:.2%} ",
                end="",
            )

        print()



    ds = master["diagnostics"]["descriptive_stats"]
    print(
        "\nDESCRIPTIVE STATISTICS "
        "(mean ± std)"
    )

    for surrogate, metrics in ds.items():

        print(f"\n{surrogate}")

        for metric_name, vals in metrics.items():
            print(
                f"  {metric_name:<25}: "
                f"{vals['mean']:.4f} ± "
                f"{vals['std']:.4f}"
            )



def print_ie_summary(master):
    seed_results = master["seed_results"]
    print("\n===== IE SUMMARY =====")

    for decoder in ["linear", "quadratic"]:
        print(f"\n--- {decoder.upper()} ---")

        stats = defaultdict(lambda: {"holds": 0, "fails": 0})

        for result in seed_results.values():
            decoder_results = result["results"][decoder]

            for name, vals in decoder_results.items():
                status = vals.get("ie_status", "").lower().strip()

                if status.startswith("ie holds"):
                    stats[name]["holds"] += 1
                elif status.startswith("ie fails"):
                    stats[name]["fails"] += 1
                else:

                    continue

        for name, c in stats.items():
            total = c["holds"] + c["fails"]
            if total == 0:
                continue

            hold_p = 100 * c["holds"] / total
            fail_p = 100 * c["fails"] / total

            print(f"{name:<18} | IE: holds {hold_p:.1f}% | fails {fail_p:.1f}%")


def print_transfer_summary(master):
    seed_results = master["seed_results"]
    print("\n===== TRANSFER SUMMARY =====")

    for decoder in ["linear", "quadratic"]:
        print(f"\n--- {decoder.upper()} ---")

        stats = defaultdict(lambda: {"holds": 0, "fails": 0})

        for result in seed_results.values():
            decoder_results = result["results"][decoder]

            for name, vals in decoder_results.items():
                status = vals.get("transfer_status", "").lower().strip()


                if status.startswith("holds"):
                    stats[name]["holds"] += 1
                elif status.startswith("fails"):
                    stats[name]["fails"] += 1
                else:
                    continue

        for name, c in stats.items():
            total = c["holds"] + c["fails"]
            if total == 0:
                continue

            hold_p = 100 * c["holds"] / total
            fail_p = 100 * c["fails"] / total

            print(f"{name:<18} | TRANSFER: holds {hold_p:.1f}% | fails {fail_p:.1f}%")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inference-equivalence "
            "simulation runner"
        )
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="all",
        help="Task type to run",
    )

    parser.add_argument(
        "--decoder",
        choices=[
            "linear",
            "quadratic",
            "both",
        ],
        default="both",
        help="Decoder family to evaluate",
    )

    return parser.parse_args()


def resolve_tasks(task_arg):
    if task_arg == "all":
        return TASK_TYPE

    if task_arg in TASK_TYPE:
        return [task_arg]

    print(
        f"[WARNING] Unknown task "
        f"'{task_arg}'. "
        f"Falling back to all "
        f"configured tasks."
    )
    return TASK_TYPE


def resolve_decoders(decoder_arg):
    if decoder_arg == "both":
        return ["linear", "quadratic"]

    return [decoder_arg]


def main():
    args = parse_args()

    tasks_to_run = resolve_tasks(
        args.type
    )
    decoders = resolve_decoders(
        args.decoder
    )

    for task_name in tasks_to_run:
        print("\n" + "=" * 60)
        print(
            f"RUNNING TASK: "
            f"{task_name.upper()}"
        )
        print(
            f"DECODER: "
            f"{', '.join(decoders).upper()}"
        )
        print("=" * 60)

        master_data = run(
            task_type=task_name,
            decoder_types=decoders,
            save_results=True,
            verbose=True,
        )

        print_summary(master_data)

        print_ie_summary(master_data)
        print_transfer_summary(master_data)


if __name__ == "__main__":
    main()
