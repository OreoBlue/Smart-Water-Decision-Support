# ICONFEST Smart Water Pipeline — V2

## What changed
V2 separates four ideas that were previously mixed together:

1. **Demand forecasting** — 1–12 month residential GPCD forecast.
2. **Supply-side forecasting** — potable supply per-capita-day using the benchmark-selected seasonal-naive method.
3. **Demand/supply pressure proxy** — residential forecast divided by potable supply forecast, with empirical Low / Watch / High bands.
4. **Administrative shortage classification** — DWR shortage stage grouped into 3 classes, explicitly treated as a secondary signal rather than physical availability.

This makes the project much easier to defend in front of judges: the model no longer calls residential consumption itself “water availability.”

## Why supply is not forced into ML
On the 2025+ holdout, a HistGradientBoosting supply model was tested and produced ~13.32% MAPE, while the 12-month seasonal-naive benchmark was ~11.90%. V2 therefore keeps the simpler benchmark. Model selection is based on validation performance, not algorithm complexity.

## Leakage / persistence warning
DWR shortage labels are highly persistent. A carry-forward label can exceed 98% accuracy in the 2025–2026 period. Therefore, future-stage accuracy by itself is a misleading headline metric. V2 keeps the DWR classifier as a secondary administrative signal and surfaces the persistence audit in `metrics.json`.

## Run
```bash
pip install -r requirements.txt
python train.py path/to/monthly_combined_dataset.csv
streamlit run app.py
```

## Competition-safe positioning
Recommended claim:

> A transferable smart-water decision-support framework benchmarked on California urban water-supplier data, combining demand forecasting, supply benchmarking, uncertainty, demand-supply pressure, and scenario analysis.

Avoid claiming that the current model is trained globally or directly forecasts reservoir availability.
