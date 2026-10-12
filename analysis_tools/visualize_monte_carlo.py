import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def visualize(csv_path, output=None):
    df = pd.read_csv(csv_path)

    # Exclude the baseline and unsuccessful/flagged runs.
    df = df.loc[
        (df["run_id"] != 0) & (df["status"] == "BEST")
    ].copy()

    if df.empty:
        raise ValueError("No randomized trials with status BEST were found.")

    metrics = {
        "price": "regional_price_load_weighted",
        "builds": "capacity_builds_regional_total",
        "retirements": "capacity_retirements_regional_total",
    }

    missing = set(metrics.values()) - set(df["metric"])
    if missing:
        raise ValueError(f"Missing required metrics: {sorted(missing)}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    # One price line per region: median across trials in each year.
    prices = (
        df.loc[df["metric"] == metrics["price"]]
        .groupby(["year", "region"])["value"]
        .median()
        .unstack("region")
        .sort_index()
    )

    for region in prices.columns:
        axes[0].plot(
            prices.index,
            prices[region],
            marker="o",
            linewidth=1.8,
            label=f"Region {region:g}",
        )

    axes[0].set(
        title="Regional electricity prices — median across trials",
        xlabel="Year",
        ylabel="Load-weighted price (native model units)",
    )
    axes[0].set_xticks(prices.index)
    axes[0].set_xticklabels([str(int(y)) for y in prices.index])
    axes[0].legend(title="Region", ncol=3, frameon=False)

    # Sum regional totals WITHIN each trial/year before making box plots.
    for ax, metric, title, color in [
        (
            axes[1],
            metrics["builds"],
            "Capacity expansion by year — all modeled regions",
            "#60a5fa",
        ),
        (
            axes[2],
            metrics["retirements"],
            "Capacity retirements by year — all modeled regions",
            "#fbbf24",
        ),
    ]:
        totals = (
            df.loc[df["metric"] == metric]
            .groupby(["run_id", "year"])["value"]
            .sum(min_count=1)
            .unstack("year")
            .sort_index(axis=1)
        )

        if totals.isna().any().any():
            raise ValueError(f"Incomplete trial/year data for {metric}")

        years = totals.columns
        samples = [totals[year].to_numpy() for year in years]

        boxes = ax.boxplot(
            samples,
            positions=range(1, len(years) + 1),
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )

        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.65)

        ax.set_xticks(range(1, len(years) + 1))
        ax.set_xticklabels([str(int(year)) for year in years])
        ax.set(title=title, xlabel="Year", ylabel="Capacity (GW)")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    n = df["run_id"].nunique()
    fig.suptitle(
        f"Demand-growth Monte Carlo — {n} randomized trials",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to results.csv")
    parser.add_argument("--output", help="Optional image output path")
    args = parser.parse_args()

    visualize(args.csv, args.output)