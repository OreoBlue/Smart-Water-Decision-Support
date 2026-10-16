# Benchmark notes

## Demand
Original HistGradientBoosting direct multi-horizon model remains a strong baseline. Extra lag features and LightGBM improved some long horizons but did not improve the overall holdout consistently enough to justify replacing the simpler model in the deployment pipeline.

## Supply
Supply was normalized as:

`Supply PCD = POTABLE_SUPPLY_GAL / (population × days_in_month)`

A quality filter of 20–1,000 gallons/capita/day retains about 99.6% of non-null supply observations while removing extreme unit/data anomalies.

Holdout benchmark (2025+):
- HistGradientBoosting supply model: ~13.32% MAPE
- Seasonal-naive same-month-last-year: ~11.90% MAPE

Selected: **seasonal naive**.

## DWR label persistence
The grouped DWR shortage stage changes very rarely in the 2025–2026 holdout. This makes naive carry-forward classification extremely strong, so future-stage accuracy can be deceptive. Do not present a high future-stage accuracy without a persistence baseline and transition-specific metrics.
