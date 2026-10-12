"""Reproducible year-level uncertainty in growth above base demand.

One normal multiplier per calendar year is shared by every region, end use,
and hour. Its standard deviation is randomize / 100. Calendar-year seeding
keeps draws stable when rows, regions or the modeled horizon are reordered.
"""

import numpy as np


def growth_multipliers(years, randomize=None, seed=None):
    """Return year -> N(1, (randomize/100)**2), without clipping or redrawing.

    These multiply the *growth increment*, not total load. A zero increment
    stays zero, including the base year. Missing/zero randomize returns ones.
    With seed=None a new realization is created at each call.
    """
    year_values = np.asarray(list(years), dtype=float)
    if not np.all(np.isfinite(year_values)) or np.any(year_values < 0):
        raise ValueError('Years must be finite, nonnegative integers.')
    if np.any(year_values != np.floor(year_values)):
        raise ValueError('Years must be integers.')
    unique_years = sorted(set(int(y) for y in year_values))
    if randomize is None:
        return dict.fromkeys(unique_years, 1.0)
    sd = float(randomize) / 100.0
    if not np.isfinite(sd) or sd < 0:
        raise ValueError('randomize must be a finite nonnegative SD percentage or None.')
    if sd == 0:
        return dict.fromkeys(unique_years, 1.0)
    root_seed = np.random.SeedSequence(seed).entropy
    return {
        y: float(np.random.default_rng(np.random.SeedSequence([root_seed, y])).normal(1.0, sd))
        for y in unique_years
    }


def randomize_annual_scalars(scalars, randomize=None, seed=None):
    """Perturb scalar minus one; retain the original dataframe when disabled."""
    factors = growth_multipliers(scalars['year'], randomize, seed)
    if randomize is None or float(randomize) == 0:
        return scalars
    result = scalars.copy()
    result['scalar'] = 1.0 + (result['scalar'] - 1.0) * result['year'].map(factors)
    if not np.isfinite(result['scalar']).all() or (result['scalar'] < 0).any():
        raise ValueError('Growth draw produced an invalid demand multiplier; not clipped/redrawn.')
    return result


def randomize_enduse_increments(increments, randomize=None, seed=None):
    """Multiply prescribed end-use growth increments by a common yearly draw."""
    factors = growth_multipliers(increments['year'], randomize, seed)
    if randomize is None or float(randomize) == 0:
        return increments
    result = increments.copy()
    result['increment_annual'] = result['increment_annual'] * result['year'].map(factors)
    if not np.isfinite(result['increment_annual']).all():
        raise ValueError('Growth draw produced nonfinite end-use increments.')
    return result
