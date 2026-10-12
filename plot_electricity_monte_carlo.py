"""Plot Monte Carlo distributions from results.csv, excluding non-BEST runs by default."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--region', required=True)
    parser.add_argument('--output', type=Path, default=Path('monte_carlo.png'))
    parser.add_argument(
        '--include-flagged',
        action='store_true',
        help='Include PENALTY and USABLE results; not recommended for baseline analysis.',
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv, dtype={'region': str, 'tech': str, 'seed': str})
    selected = df[df['region'].eq(args.region)].copy()
    all_runs = selected['run_id'].nunique()
    if not args.include_flagged:
        selected = selected[selected['status'].eq('BEST')]
    print(
        f'Included {selected["run_id"].nunique()} of {all_runs} exported runs for region {args.region}. '
        'Check runs.csv separately for failed trials.'
    )
    selected['year'] = pd.to_numeric(selected['year'], errors='raise').astype(int)
    metrics = [
        ('regional_price_load_weighted', 'Load-weighted regional price', 'Native model price'),
        ('capacity_builds_regional_total', 'New builds in each modeled year', 'GW'),
        ('capacity_retirements_regional_total', 'Retirements in each modeled year', 'GW'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), layout='constrained')
    for ax, (metric, title, unit) in zip(axes, metrics):
        subset = selected[selected['metric'].eq(metric)]
        trials = subset[subset['run_id'].gt(0)]
        if trials.empty:
            ax.text(0.5, 0.5, 'No accepted trial data', ha='center', transform=ax.transAxes)
        else:
            stats = trials.groupby('year')['value'].agg(
                median='median', p05=lambda x: x.quantile(0.05), p95=lambda x: x.quantile(0.95)
            )
            ax.fill_between(
                stats.index, stats['p05'], stats['p95'], alpha=0.2, label='5–95% of trials'
            )
            ax.plot(stats.index, stats['median'], marker='o', label='Trial median')
        baseline = subset[subset['run_id'].eq(0)].sort_values('year')
        if not baseline.empty:
            ax.plot(
                baseline['year'],
                baseline['value'],
                '--',
                color='black',
                label='Unrandomized baseline',
            )
        ax.set(title=title, xlabel='Model year', ylabel=unit)
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle(f'Region {args.region}: demand-growth Monte Carlo')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
