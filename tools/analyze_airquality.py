"""Fit the documented exploratory ozone model from the bundled R dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

import kernelmix as km

DATA_SHA256 = "2c30fd88f946fb033340b1058465fcf791944d031d3f1c6d653515b7be5a74b3"
EXPOSURES = ["Solar.R", "Wind", "Temp"]


def load_data(path):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != DATA_SHA256:
        raise ValueError("data differs from the documented R 4.5.0 airquality export")
    rows = list(csv.DictReader(content.decode("utf-8").splitlines()))
    values = np.array([[np.nan if row[k] == "NA" else float(row[k]) for k in
                        ["Ozone", *EXPOSURES, "Month", "Day"]] for row in rows])
    keep = np.isfinite(values[:, :4]).all(axis=1)
    retained = values[keep]
    if (retained[:, 0] <= 0).any():
        raise ValueError("ozone must be positive before taking logarithms")
    x = (retained[:, 4, None] == np.arange(6, 10)).astype(float)
    if np.linalg.matrix_rank(np.column_stack((np.ones(len(x)), x))) != 5:
        raise ValueError("month adjustment design is not full rank")
    counts = [{"month": month, "recorded": int(np.sum(values[:, 4] == month)),
               "complete": int(np.sum(retained[:, 4] == month))} for month in range(5, 10)]
    profile = {"recorded": len(rows), "complete": int(keep.sum()),
               "excluded_rows": (np.flatnonzero(~keep) + 1).tolist(), "month_counts": counts,
               "missing_by_field": {name: int(np.isnan(values[:, j]).sum()) for j, name in
                                    enumerate(["Ozone", *EXPOSURES, "Month", "Day"])}}
    return retained, x, keep, values, profile


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] /
                        "docs/data/airquality.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--archive", type=Path, help="Destination for a portable fitted-model archive")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output directory to preserve previous results")
    args.output.mkdir(parents=True)
    values, x, keep, original, profile = load_data(args.data)
    settings = {"draws": args.draws, "warmup": args.warmup, "chains": args.chains,
                "seed": args.seed, "family": "gaussian", "selection": "none",
                "prior": {"r_distribution": "gamma", "r_mean": 1., "r_sd": 1.,
                          "sigma2_shape": 2., "sigma2_rate": 1.}}
    started = time.perf_counter()
    fit = km.fit(np.log(values[:, 0]), values[:, 1:4], x,
                 exposure_names=EXPOSURES,
                 covariate_names=["June", "July", "August", "September"], **settings)
    fit_seconds = time.perf_counter() - started
    if args.archive is not None:
        fit.save(args.archive)
    (args.output / "summary.txt").write_text(fit.summary() + "\n", encoding="utf-8")
    diag = fit.diagnostics()
    write_csv(args.output / "parameters.csv", fit.summary_table())
    write_csv(args.output / "diagnostics.csv", diag["parameters"])
    # Small joint grids retain covariance while avoiding a full grid covariance.
    curve_rows = []
    for j, name in enumerate(EXPOSURES):
        grid = np.unique(np.quantile(values[:, j + 1], np.linspace(.05, .95, 21)))
        curve = km.univariate_response(fit, variable=j, grid=grid, max_draws=400, seed=89)
        for i in range(len(curve.mean)):
            profile_row = np.median(values[:, 1:4], axis=0)
            profile_row[j] = curve.locations[i]["value"]
            distances = np.linalg.norm((values[:, 1:4] - profile_row) /
                                       values[:, 1:4].std(axis=0, ddof=1), axis=1)
            curve_rows.append({"variable": name, "value": float(curve.locations[i]["value"]),
                               "nearest_standardized_distance": float(distances.min()),
                               "mean": float(curve.mean[i]), "lower": float(curve.lower[i]),
                               "upper": float(curve.upper[i])})
    write_csv(args.output / "curves.csv", curve_rows)
    mean_h = fit.posterior_moments_h(max_draws=400)["mean"]
    selected = np.linspace(0, fit.settings["draws"] - 1,
                           min(fit.settings["draws"], 400), dtype=int)
    beta = fit.samples["beta"][:, selected].mean(axis=(0, 1))
    fitted = mean_h + fit.data["X"] @ beta
    residual = np.log(values[:, 0]) - fitted
    dates = np.array([np.datetime64(f"1973-{int(m):02d}-{int(d):02d}") for m, d in
                      zip(values[:, 4], values[:, 5])])
    adjacent = np.diff(dates).astype("timedelta64[D]").astype(int) == 1
    lag_correlation = float(np.corrcoef(residual[:-1][adjacent], residual[1:][adjacent])[0, 1])
    write_csv(args.output / "observations.csv", ({"source_row": int(row), "date": str(date),
              "ozone_ppb": float(v[0]), "log_ozone": float(np.log(v[0])),
              "fitted_log_ozone": float(mu), "residual": float(err)} for row, date, v, mu, err in
              zip(np.flatnonzero(keep) + 1, dates, values, fitted, residual)))
    finite_ozone = np.isfinite(original[:, 0])
    missing_exposure = ~np.isfinite(original[:, 1:4]).all(axis=1)
    missing_comparison = {}
    for label, subset in [("complete", keep), ("observed_ozone_missing_exposure",
                                               finite_ozone & missing_exposure)]:
        observed = original[subset, 0]
        missing_comparison[label] = {"n": len(observed), "median_ozone_ppb":
                                     float(np.median(observed)) if len(observed) else None}
    record = {"data_sha256": DATA_SHA256, "settings": settings, "profile": profile,
              "missing_comparison": missing_comparison, "fit_seconds": fit_seconds,
              "total_seconds": time.perf_counter() - started,
              "consecutive_day_pairs": int(adjacent.sum()),
              "consecutive_day_residual_correlation": lag_correlation,
              "diagnostic_messages": diag["messages"], "sampling_info": fit.sampling_info,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "kernelmix": km.__version__, "numpy": np.__version__,
                              "scipy": scipy.__version__},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "Exploratory complete-case associations under independent Gaussian working errors; no causal, forecasting, general calibration or original-study reproduction claim."}
    (args.output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False),
                                              encoding="utf-8")
    print(json.dumps({"rows": len(values), "fit_seconds": fit_seconds,
                      "diagnostic_messages": diag["messages"]}))


if __name__ == "__main__":
    main()
