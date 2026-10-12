"""Tests for growth-only sampling and time-weighted Monte Carlo reporting."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from src.common.demand_growth import growth_multipliers, randomize_annual_scalars
from src.common.utilities import scale_load, scale_load_with_enduses
from run_electricity_monte_carlo import extract_results


@pytest.fixture
def data(tmp_path):
    pd.DataFrame(
        {'region': ['A', 'A', 'B', 'B'], 'hour': [1, 2, 1, 2], 'Load': [100.0, 300.0, 200.0, 600.0]}
    ).to_csv(tmp_path / 'BaseLoad.csv', index=False)
    pd.DataFrame({'year': [2023, 2030], 'scalar': [1.0, 1.2]}).to_csv(
        tmp_path / 'LoadScalar.csv', index=False
    )
    pd.DataFrame({'enduse_cat': ['heat', 'other'], 'base_year_share': [0.4, 0.6]}).to_csv(
        tmp_path / 'EnduseBaseShares.csv', index=False
    )
    pd.DataFrame(
        {
            'year': [2023, 2023, 2030, 2030],
            'enduse_cat': ['heat', 'other'] * 2,
            'increment_annual': [0.0, 0.0, 0.3, 0.1],
        }
    ).to_csv(tmp_path / 'EnduseScalar.csv', index=False)
    pd.DataFrame(
        {
            'enduse_cat': ['heat', 'heat', 'other', 'other'],
            'hour': [1, 2, 1, 2],
            'share': [0.75, 0.25, 0.25, 0.75],
        }
    ).to_csv(tmp_path / 'EnduseShapes.csv', index=False)
    return tmp_path


@pytest.mark.parametrize('mode', ['annual', 'end_use'])
def test_disabled_seed_and_shared_growth(data, mode):
    fn = (
        scale_load
        if mode == 'annual'
        else lambda p, **kw: scale_load_with_enduses(p, ['A', 'B'], **kw)
    )
    baseline = fn(data)
    pd.testing.assert_frame_equal(baseline, fn(data, randomize=0, seed=9))
    pd.testing.assert_frame_equal(baseline, fn(data, randomize=None, seed=9))
    first = fn(data, randomize=10, seed=12)
    pd.testing.assert_frame_equal(first, fn(data, randomize=10, seed=12))
    assert not first.equals(fn(data, randomize=10, seed=13))
    pd.testing.assert_frame_equal(first[first.year == 2023], baseline[baseline.year == 2023])
    base = baseline[baseline.year == 2023].set_index(['region', 'hour'])['Load']
    original_growth = baseline[baseline.year == 2030].set_index(['region', 'hour'])['Load'] - base
    sampled_growth = first[first.year == 2030].set_index(['region', 'hour'])['Load'] - base
    # Same fractional change to growth at every hour/region, not independent hourly noise.
    np.testing.assert_allclose(
        sampled_growth / original_growth, growth_multipliers([2030], 10, 12)[2030], atol=1e-4
    )


def test_normal_moments_and_year_stability():
    values = np.array([growth_multipliers([2030], 10, s)[2030] for s in range(10000)])
    assert abs(values.mean() - 1) < 0.004
    assert abs(values.std() - 0.1) < 0.004
    a = growth_multipliers([2030, 2040], 10, 17)
    assert a == growth_multipliers([2040, 2030], 10, 17)
    assert a[2030] == growth_multipliers([2030], 10, 17)[2030]
    assert a[2030] != a[2040]


@pytest.mark.parametrize('bad', [-1, np.nan, np.inf])
def test_bad_sd(bad):
    with pytest.raises(ValueError):
        growth_multipliers([2030], bad, 1)


def test_no_clipping():
    scalars = pd.DataFrame({'year': [2030], 'scalar': [3.0]})
    # Find a fixed seed whose normal growth shock produces a negative total multiplier.
    seed = next(s for s in range(100) if growth_multipliers([2030], 1000, s)[2030] < -0.5)
    with pytest.raises(ValueError, match='not clipped'):
        randomize_annual_scalars(scalars, 1000, seed)


def test_price_export_weights_and_sign():
    m = pyo.ConcreteModel()
    m.idx = pyo.Set(initialize=[('A', 2030, 1), ('A', 2030, 2)], dimen=3)
    m.map_hour_day = {1: 1, 2: 2}
    m.weight_day = {1: 2.0, 2: 3.0}
    m.weight_year = {2030: 5.0}
    m.weight_hour = {1: 1.0, 2: 4.0}
    m.elec_load = pyo.Param(m.idx, initialize={('A', 2030, 1): 2.0, ('A', 2030, 2): 8.0})
    m.gen = pyo.Var(m.idx, within=pyo.NonNegativeReals)
    m.unmet_load = pyo.Var(m.idx, within=pyo.NonNegativeReals, bounds=(0, 0))
    m.demand_balance = pyo.Constraint(
        m.idx, rule=lambda m, r, y, h: m.gen[r, y, h] >= m.elec_load[r, y, h]
    )
    m.total_cost = pyo.Objective(
        expr=sum(
            m.gen[r, y, h]
            * m.weight_day[m.map_hour_day[h]]
            * m.weight_year[y]
            * {1: 10.0, 2: 20.0}[h]
            for r, y, h in m.idx
        )
    )
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory('highs').solve(m)
    for v in m.unmet_load.values():
        v.set_value(0.0)
    rows = extract_results(m)
    segment = rows[rows.metric == 'regional_price_segment']
    np.testing.assert_allclose(segment.value, [10.0, 20.0])
    time_price = rows.loc[rows.metric == 'regional_price_time_weighted', 'value'].item()
    assert time_price == pytest.approx((10 * 2 + 20 * 12) / 14)
    load_price = rows.loc[rows.metric == 'regional_price_load_weighted', 'value'].item()
    assert load_price == pytest.approx((10 * 4 + 20 * 24) / 28)
    m.dual.clear()
    with pytest.raises(ValueError, match='Missing'):
        extract_results(m)


def test_expected_unrandomized_values(data):
    annual = scale_load(data).sort_values(['year', 'region', 'hour'])
    np.testing.assert_allclose(annual.Load, [100, 300, 200, 600, 120, 360, 240, 720])
    enduse = scale_load_with_enduses(data, ['A', 'B']).sort_values(['year', 'region', 'hour'])
    np.testing.assert_allclose(enduse.Load, [100, 300, 200, 600, 142, 330, 284, 660])
