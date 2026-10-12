# Demand-growth uncertainty and Monte Carlo

Based on Community-NEMS/cnems-models commit
`f0f8d791bf0170abefb2a5cfa6691a7a1299884f`, reviewed October 12, 2026.

## Statistical definition

`randomize=None` or 0 preserves the original scaling. Positive `randomize=p`
means a standard deviation of p percent of **growth above base demand**, not
p percent of total demand and not p percentage points of growth.

For each trial and calendar year y, draw Z_y ~ Normal(1, (p/100)^2).

Annual mode:

    sampled_scalar_y = 1 + (input_scalar_y - 1) * Z_y
    load[r,y,h] = round(base_load[r,h] * sampled_scalar_y, 3)

End-use mode:

    sampled_increment[e,y] = input_increment[e,y] * Z_y

All subsequent base-share and hourly-shape calculations remain unchanged.
Example: baseline growth of 20%, randomize=10 gives mean growth 20% and SD
2 percentage points. The total multiplier has mean 1.20 and SD 0.02.

The SAME year shock applies across all hours, regions and end uses. There is
no independent hourly noise. Annual mode preserves hourly shape up to the
existing rounding. End-use mode changes the size of the prescribed growth
profile relative to the fixed base profile, but introduces no hourly randomness.

Different years have independent shocks. These perturbations apply to the
input cumulative growth levels, not to compounded year-on-year rates. Paths
need not be monotone. Perfect within-year cross-region/end-use correlation
and independent years are explicit scenario assumptions, not estimated facts.
Zero growth stays zero, preserving the shipped 2023 base year. Negative growth
has SD p/100 times its absolute magnitude. The exported Z for a zero-growth
year is immaterial.

No clipping or redrawing: invalid negative total loads cause recorded failed
trials. High SD can cause such failures. Inspect runs.csv before interpreting
statistics conditional on successful trials. Normal growth itself can reverse
sign without being invalid if the resulting demand stays nonnegative.

Sampling precedes temporal aggregation. Averaging hours does not cancel a
shared year shock; averaging multiple years can average independent shocks.
Use aggregate_years=false for direct selected-year comparisons.

## Install

The ZIP contains changes.patch and complete files under files/. Extract it
outside your repository. From your repository root:

    git apply --check /path/to/cnems_demand_growth_mc/changes.patch
    git apply /path/to/cnems_demand_growth_mc/changes.patch

If --check fails, stop and apply the edits manually using files/ as a reference.
Do not overwrite unrelated edits or force the patch. This patch targets the
ORIGINAL repository, not the previous hourly-noise proposal. If you implemented
that proposal, remove its randomize_load helper/calls and use this growth-only
implementation, retaining only one copy of the config fields.

No new dependencies are added to the repository's existing Pixi environment.

## Exact files changed

- NEW src/common/demand_growth.py: seeded yearly draws and scalar transformations.
- src/common/utilities.py: imports these helpers; adds randomize=None and seed=None
  to scale_load and scale_load_with_enduses; modifies scalar tables BEFORE use.
- src/models/electricity/elec_config.py: imports Field; adds randomize (finite,
  nonnegative, default None) and random_seed (nonnegative, default None).
- src/models/electricity/param_data.py: passes both settings in both branches of
  ParamData.build_load_dataframe. These are the two direct scaling callers.
- run_configs/basic_elec_config.toml: optional settings shown as comments.
- NEW run_configs/monte_carlo_elec_config.toml: basic end-use scenario with expansion
  enabled; retains regions 7/8/9 and years 2025/2030. Edit for a larger study.
- NEW run_electricity_monte_carlo.py: runner, extraction, checkpoints and resume.
- NEW plot_electricity_monte_carlo.py: plots from the tidy CSV.
- NEW tests/test_demand_growth_mc.py: statistical and price extraction checks.
- This documentation.

For a regular model run, add under [elec_config]:

    randomize = 10.0
    random_seed = 123

Omit randomize to disable it; TOML has no None. A fixed seed reuses the same
realization on every rebuild. Always set a seed when iterating coupled models.
Without a seed, each scaling call generates a new realization.

## Monte Carlo command

From the repository root:

    pixi run python run_electricity_monte_carlo.py \
      --config run_configs/monte_carlo_elec_config.toml \
      --runs 100 --randomize 10 --master-seed 123 \
      --output output/growth_mc_100

Runs one unrandomized baseline (run_id=0) plus 100 trials. The runner deliberately
accepts ONLY standalone electricity configurations and requires capacity_expansion=true.
It honors other config settings. The command-line SD and generated trial seeds
override the corresponding TOML values for this runner only.

Each trial builds a fresh model and uses the existing ElectricitySequencer to
solve, including its learning loop. Demand is sampled once during construction
and stays fixed for every solve within a trial. Different run IDs get different
SeedSequence child seeds. Extending the experiment preserves all earlier seeds.

Resume by repeating the command with --resume. You may increase --runs. Settings,
input/code hashes and environment versions must match. Paths are part of the
fingerprint, so moving to another checkout requires a new output directory.
An interrupted trial without its JSON completion marker is rerun with its original
seed. Completed failed trials are retained, not silently retried. A failed baseline
stops the experiment: fix the cause and start a new output directory.

One .running.lock prevents concurrent writers. After a hard process kill, remove
it only after verifying no runner is active. Completed trial CSVs and JSON markers
are written atomically; the combined CSV is rebuilt from completed checkpoints,
so resume does not duplicate trials.

## CSV contents

results.csv is tidy long form with columns:

    run_id, seed, randomize_sd_pct, status, region, year, hour,
    tech, step, metric, value, unit

Metrics:

- regional_price_segment: price at each time segment.
- regional_price_time_weighted: annual average using day_weight * hour_weight.
- regional_price_load_weighted: annual average using day_weight * segment load.
- capacity_builds, capacity_retirements, capacity_total: region-tech-step-year GW.
- capacity_builds_regional_total, capacity_retirements_regional_total,
  capacity_total_regional_total: technology/step sums within a region and year.
- annual_demand, annual_unmet_load: representative annual energy, GWh.
- growth_multiplier: actual Z_y for every input calendar year.
- objective: native model objective units.

Builds/retirements are decisions in the indicated model year, not cumulative
capacity or rates per intervening year. Retirements are positive. Missing indices
are not fabricated as zeros. Regional totals sum the represented indices only.

runs.csv records every trial's seed, status, seconds, error and nonlinear flag.
manifest.json records provenance. runs/ stores per-trial CSV/JSON checkpoints,
logs, and failure tracebacks.

## Price and status interpretation

Prices are demand-balance duals divided by day and year weights, matching the
repository's get_elec_price convention. No additional hour-weight division is
needed because balances are in segment energy. Annual averages DO account for
unequal segment durations. Price signs are preserved; missing/nonfinite duals
cause recorded errors, never invented zero prices.

Prices retain the native SupplyPrice/objective scale, labeled model_price. The
upstream schema does not consistently establish currency scaling, so this package
does not impose an unverified dollars/MWh conversion. These are balance duals,
not retail prices or the total derivative of cost with respect to all simultaneous
load-dependent reserve requirements.

BEST: accepted solution without material unmet load.
USABLE: linear-learning loop reached its cap without convergence.
PENALTY: solved with material unmet load.
ERROR: invalid sample, failed solve or incomplete export.

The runner exits with code 2 if any trial is not BEST. Flagged solved results
remain in the CSV with status. Plots include BEST only by default; check runs.csv
for failed/flagged rates first. Nonlinear learning uses the repo's IPOPT path:
BEST is a sequencer status, not a global-optimality certificate for nonlinear runs.

Each trial's full future demand path is known to the optimizer. This is Monte
Carlo sensitivity of scenario-specific perfect-foresight investment plans, not
a stochastic program with common first-stage decisions or adaptive foresight.

## Plot

    pixi run python plot_electricity_monte_carlo.py \
      output/growth_mc_100/results.csv --region 7 \
      --output output/growth_mc_100/region_7.png

Three panels show load-weighted annual price, builds, and retirements: baseline,
trial median and 5th-95th percentile outcome band. The band is not a confidence
interval for an estimated mean. Use results.csv for technology-specific graphs.

## Verification

    pixi run python -m pytest -o addopts='' tests/test_demand_growth_mc.py -q

Sampling checks cover disabled behavior, exact baseline calculations, common
annual shocks, fixed seeds, year/region independence of seed selection, normal
moments, and invalid-input handling. A real small HiGHS model verifies price
signs, de-weighting, annual averaging and missing-dual rejection.

Also exercised: real annual-mode linear-learning trials, real end-use-mode
expansion trials, CSV plotting, resume/extension and duplicate prevention.
These are smoke checks, not a calibration or economic validation of the model.
