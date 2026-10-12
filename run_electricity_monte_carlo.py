"""Run standalone electricity demand-growth scenarios; checkpoint every trial.

Run from the repository root with its Pixi environment. See
 docs/demand_growth_monte_carlo.md for definitions, units and resume semantics.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from definitions import PROJECT_ROOT
from src.common.common_config import parse_config_file
from src.common.demand_growth import growth_multipliers
from src.common.integrated_model_sequencer import IterationStatus
from src.models.electricity.elec_config import ElecConfig
from src.models.electricity.sequencer import ElectricitySequencer

COLUMNS = [
    'run_id',
    'seed',
    'randomize_sd_pct',
    'status',
    'region',
    'year',
    'hour',
    'tech',
    'step',
    'metric',
    'value',
    'unit',
]


def extract_results(model):
    """Return tidy native-unit results; never replace absent duals with zero."""
    records = []

    def add(metric, value, unit, region='', year='', hour='', tech='', step=''):
        numeric = float(pyo.value(value))
        if not math.isfinite(numeric):
            raise ValueError(f'Nonfinite {metric} at {region}, {year}, {hour}.')
        records.append(
            dict(
                region=str(region),
                year=year,
                hour=hour,
                tech=str(tech),
                step=step,
                metric=metric,
                value=numeric,
                unit=unit,
            )
        )

    regional = {}
    for (r, y, h), constraint in model.demand_balance.items():
        if constraint not in model.dual:
            raise ValueError(f'Missing demand-balance dual at {(r, y, h)}.')
        day_weight = float(pyo.value(model.weight_day[model.map_hour_day[h]]))
        hour_weight = float(pyo.value(model.weight_hour[h]))
        year_weight = float(pyo.value(model.weight_year[y]))
        if min(day_weight, hour_weight, year_weight) <= 0:
            raise ValueError('Price de-weighting requires positive time weights.')
        # Matches get_elec_price: divide by day and year weights, not hour weight.
        price = float(model.dual[constraint]) / (day_weight * year_weight)
        energy = float(pyo.value(model.elec_load[r, y, h])) * day_weight
        unmet = float(pyo.value(model.unmet_load[r, y, h])) * day_weight
        duration = day_weight * hour_weight
        add('regional_price_segment', price, 'model_price', r, y, h)
        totals = regional.setdefault((r, y), np.zeros(5))
        totals += [price * duration, duration, price * energy, energy, unmet]
    for (r, y), (pt, duration, pe, energy, unmet) in regional.items():
        add('regional_price_time_weighted', pt / duration, 'model_price', r, y)
        if energy > 0:
            add('regional_price_load_weighted', pe / energy, 'model_price', r, y)
        add('annual_demand', energy, 'GWh', r, y)
        add('annual_unmet_load', unmet, 'GWh', r, y)

    for name in ('capacity_builds', 'capacity_retirements', 'capacity_total'):
        component = getattr(model, name, None)
        if component is None:
            continue
        totals = {}
        for (r, tech, step, y), var in component.items():
            amount = float(pyo.value(var))
            add(name, amount, 'GW', r, y, tech=tech, step=step)
            totals[r, y] = totals.get((r, y), 0.0) + amount
        for (r, y), amount in totals.items():
            add(name + '_regional_total', amount, 'GW', r, y)
    add('objective', model.total_cost, 'model_cost')
    return pd.DataFrame(records)


def atomic_csv(frame, path):
    """Publish a complete CSV via an atomic replacement on the same filesystem."""
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def atomic_json(data, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    temp.replace(path)


def fingerprint(common, elec, config_path):
    """Detect changed inputs/code/packages before resuming an experiment."""
    files = {
        config_path.resolve(),
        Path(__file__).resolve(),
        PROJECT_ROOT / 'definitions.py',
        PROJECT_ROOT / 'pyproject.toml',
    }
    for directory in (
        PROJECT_ROOT / 'src',
        PROJECT_ROOT / common.common_data_path,
        common.residential_data_path,
        elec.input_path,
        PROJECT_ROOT / 'input/integrator',
    ):
        files.update(
            p
            for p in directory.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'}
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path).encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    packages = {
        name: importlib.metadata.version(name)
        for name in ('numpy', 'pandas', 'pyomo', 'pydantic', 'highspy')
    }
    return dict(sha256=digest.hexdigest(), packages=packages, python=sys.version)


def combine(output):
    """Completed per-trial files are authoritative, so resume cannot duplicate rows."""
    runs = output / 'runs'
    outcomes = [json.loads(p.read_text()) for p in sorted(runs.glob('run_*.json'))]
    atomic_csv(pd.DataFrame(outcomes), output / 'runs.csv')
    destination = output / 'results.csv'
    temp = destination.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8', newline='') as target:
        pd.DataFrame(columns=COLUMNS).to_csv(target, index=False)
        for outcome in outcomes:
            if outcome['status'] == 'ERROR':
                continue
            source = runs / f'run_{outcome["run_id"]:06d}.csv'
            if not source.exists():
                raise RuntimeError(f'Completed trial is missing its results: {source}')
            with source.open(encoding='utf-8') as stream:
                next(stream)  # skip header
                for line in stream:
                    target.write(line)
    temp.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=100, help='Random trials, plus baseline run 0.')
    parser.add_argument('--randomize', type=float, default=10.0, help='SD percent of growth.')
    parser.add_argument('--master-seed', type=int, default=123)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (
        args.runs < 1
        or args.master_seed < 0
        or not math.isfinite(args.randomize)
        or args.randomize < 0
    ):
        parser.error('runs >= 1, master-seed >= 0, and finite randomize >= 0 are required.')
    common, remainder = parse_config_file(args.config)
    elec = ElecConfig(**remainder['elec_config'])
    if common.mode.value != 'standalone' or [m.value for m in common.models_to_run] != [
        'electricity'
    ]:
        parser.error('Use a standalone config with models_to_run=["electricity"].')
    if not elec.capacity_expansion:
        parser.error('Set capacity_expansion=true to measure builds and retirements.')
    output = args.output.resolve()
    expected = dict(
        format_version=1,
        randomize_sd_pct=args.randomize,
        master_seed=args.master_seed,
        config=str(args.config.resolve()),
        provenance=fingerprint(common, elec, args.config),
    )
    if args.resume:
        manifest = output / 'manifest.json'
        if not manifest.exists() or json.loads(manifest.read_text()) != expected:
            parser.error(
                'Resume manifest differs: keep the same settings, inputs, code and environment.'
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(expected, output / 'manifest.json')
    if args.resume:
        completed = list((output / 'runs').glob('run_*.json'))
        if completed and max(int(p.stem.split('_')[1]) for p in completed) > args.runs:
            parser.error('Cannot resume with fewer trials than already completed.')
        baseline_marker = output / 'runs' / 'run_000000.json'
        if (
            baseline_marker.exists()
            and json.loads(baseline_marker.read_text())['status'] == 'ERROR'
        ):
            parser.error('The baseline failed. Fix its error and start a new output directory.')
    (output / 'runs').mkdir(exist_ok=True)
    # One writer per experiment; context manager releases this lock on any Python exception.
    lock = output / '.running.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(
            f'{lock} exists; another runner may be active. Remove only after verifying it stopped.'
        )
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    try:
        for run_id in range(args.runs + 1):
            marker = output / 'runs' / f'run_{run_id:06d}.json'
            if marker.exists():
                continue
            seed = int(
                np.random.SeedSequence(args.master_seed, spawn_key=(run_id,)).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            sd = 0.0 if run_id == 0 else args.randomize
            started = time.perf_counter()
            outcome = dict(
                run_id=run_id,
                seed=str(seed),
                randomize_sd_pct=sd,
                status='ERROR',
                error='',
                seconds=0.0,
                nonlinear=elec.expansion_learning_type.value == 'nonlinear',
            )
            log_handler = logging.FileHandler(output / 'runs' / f'run_{run_id:06d}.log', mode='w')
            logging.getLogger().addHandler(log_handler)
            try:
                trial_config = ElecConfig(
                    **(elec.model_dump() | {'randomize': sd, 'random_seed': seed})
                )
                sequencer = ElectricitySequencer()
                model = sequencer.build_model(common, trial_config)
                _, status = sequencer.solve_model()
                if status is IterationStatus.ERROR:
                    raise RuntimeError('Solver reported ERROR; no solution exported.')
                frame = extract_results(model)
                source = (
                    'LoadScalar.csv'
                    if elec.load_scale_mode.value == 'annual'
                    else 'EnduseScalar.csv'
                )
                factors = growth_multipliers(
                    pd.read_csv(common.residential_data_path / source)['year'], sd, seed
                )
                shock_rows = pd.DataFrame(
                    [
                        dict(
                            region='',
                            year=y,
                            hour='',
                            tech='',
                            step='',
                            metric='growth_multiplier',
                            value=factor,
                            unit='ratio',
                        )
                        for y, factor in factors.items()
                    ]
                )
                frame = pd.concat([frame, shock_rows], ignore_index=True)
                outcome['status'] = status.name
                for key in ('run_id', 'seed', 'randomize_sd_pct', 'status'):
                    frame[key] = outcome[key]
                atomic_csv(frame[COLUMNS], output / 'runs' / f'run_{run_id:06d}.csv')
            except Exception as exc:
                outcome['status'] = 'ERROR'
                outcome['error'] = f'{type(exc).__name__}: {exc}'
                logging.exception('Trial %s failed', run_id)
                (output / 'runs' / f'run_{run_id:06d}.error.txt').write_text(traceback.format_exc())
            finally:
                logging.getLogger().removeHandler(log_handler)
                log_handler.close()
            outcome['seconds'] = time.perf_counter() - started
            atomic_json(outcome, marker)
            combine(output)
            print(
                f'run={run_id} seed={seed} status={outcome["status"]} seconds={outcome["seconds"]:.1f}',
                flush=True,
            )
            if run_id == 0 and outcome['status'] == 'ERROR':
                print('Baseline failed; fix the recorded error before running random trials.')
                return 2
            if 'model' in locals():
                del model
            if 'sequencer' in locals():
                del sequencer
            gc.collect()
        combine(output)
    finally:
        lock.unlink(missing_ok=True)
    outcomes = pd.read_csv(output / 'runs.csv')
    counts = outcomes['status'].value_counts().to_dict()
    print(f'Completed statuses: {counts}. Results: {output / "results.csv"}')
    return 0 if (outcomes['status'] == 'BEST').all() else 2


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
