"""Posterior summaries, joint predictions, and portable fitted objects."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .diagnostics import diagnose


def _probability(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("probability must be between zero and one")
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("probability must be between zero and one")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _freeze(array):
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


def _matrix(value, columns, names, name):
    if hasattr(value, "columns"):
        if list(map(str, value.columns)) != list(names):
            raise ValueError(f"{name} columns must match the training names and order")
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if result.ndim != 2 or result.shape[1] != columns or len(result) == 0:
        raise ValueError(f"{name} must have shape (rows, {columns}) with at least one row")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _normal_draw(mean, covariance, rng):
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    tolerance = 1e-9 * max(1.0, np.max(np.abs(values)))
    if values.min(initial=0) < -tolerance:
        raise np.linalg.LinAlgError("predictive covariance is materially indefinite")
    return mean + vectors @ (np.sqrt(np.maximum(values, 0)) * rng.normal(size=len(mean)))


@dataclass(frozen=True)
class Prediction:
    """Posterior predictions with intervals for each requested row.

    ``draws`` has axes (chain, saved draw, prediction row). Rows are sampled
    together so their dependence is retained. ``covariance`` is computed from
    these joint draws. When h is predicted, its uncertainty at each saved state
    is included. Mean predictions exclude new observation noise; observation predictions include it. Intervals
    have equal posterior probability below and above their limits and apply to
    each row separately.
    """

    draws: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    covariance: np.ndarray
    probability: float
    scale: str
    kind: str

    def to_records(self):
        return [
            {"row": i, "mean": float(self.mean[i]), "sd": float(self.sd[i]),
             "lower": float(self.lower[i]), "upper": float(self.upper[i])}
            for i in range(len(self.mean))
        ]


def _prediction(draws, probability, scale, kind):
    probability = _probability(probability)
    flat = np.asarray(draws).reshape(-1, draws.shape[-1])
    if len(flat) < 2 or not np.isfinite(flat).all():
        raise ValueError("prediction needs at least two finite posterior draws")
    tail = (1 - probability) / 2
    lower, upper = np.quantile(flat, [tail, 1 - tail], axis=0)
    return Prediction(
        *map(_freeze, (draws, flat.mean(axis=0), flat.std(axis=0, ddof=1), lower, upper,
                       np.atleast_2d(np.cov(flat, rowvar=False, ddof=1)))),
        probability, scale, kind,
    )


@dataclass(frozen=True)
class SurvivalPrediction:
    """Survival probabilities, hazards or cumulative incidences over time.

    ``draws`` has axes (chain, saved draw, row, time). Intervals apply to each
    row/time separately. A hazard is an event rate among those still at risk;
    a cumulative incidence is the probability of a specified cause by a time.
    """
    draws: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    times: np.ndarray
    probability: float
    kind: str

    def to_records(self):
        return [{"row": row, "time": float(time), "mean": float(self.mean[row, j]),
                 "sd": float(self.sd[row, j]), "lower": float(self.lower[row, j]),
                 "upper": float(self.upper[row, j]), "kind": self.kind}
                for row in range(self.mean.shape[0]) for j, time in enumerate(self.times)]


def _survival_prediction(draws, times, probability, kind):
    probability = _probability(probability)
    flat = draws.reshape((-1,) + draws.shape[2:])
    tail = (1-probability)/2
    lower, upper = np.quantile(flat, [tail, 1-tail], axis=0)
    return SurvivalPrediction(*map(_freeze, (draws, flat.mean(axis=0), flat.std(axis=0, ddof=1),
                                             lower, upper, times)), probability, kind)


class BKMRFit:
    """Saved BKMR chains, data transformations, summaries, and predictions.

    Fit objects retain the training data. Treat a saved object as research data
    when deciding where to store or share it. Arrays are copied and read-only;
    sample-extraction methods return independent copies.
    """

    def __init__(self, data, samples, settings, sampling_info):
        self.data = {k: _freeze(v) if isinstance(v, np.ndarray) else deepcopy(v)
                     for k, v in data.items()}
        self.samples = {k: _freeze(v) for k, v in samples.items()}
        self.settings = deepcopy(settings)
        self.sampling_info = deepcopy(sampling_info)
        self._validate()

    def _validate(self):
        required_data = {"y", "Z", "Z_original", "X", "X_original", "z_center", "z_scale",
                         "rows", "original_n", "exposure_names", "covariate_names", "groups"}
        if not required_data.issubset(self.data):
            raise ValueError("fit data are incomplete")
        required_samples = {"beta", "r", "lambda", "sigma2", "delta", "pi"}
        if not required_samples.issubset(self.samples):
            raise ValueError("posterior samples are incomplete")
        if self.family not in ("gaussian", "binomial", "poisson", "negative_binomial", "survival"):
            raise ValueError("unknown outcome family")
        y, z, x = self.data["y"], self.data["Z"], self.data["X"]
        if y.ndim != 1 or z.ndim != 2 or x.ndim != 2 or len(z) != len(y) or len(x) != len(y):
            raise ValueError("incompatible training data shapes")
        n, m, p = len(y), z.shape[1], x.shape[1]
        chains = _integer(self.settings["chains"], "chains")
        draws = _integer(self.settings["draws"], "draws", 2)
        shapes = {"beta": (chains, draws, p), "r": (chains, draws, m),
                  "lambda": (chains, draws), "sigma2": (chains, draws),
                  "delta": (chains, draws, m), "pi": (chains, draws)}
        if self.family == "binomial":
            shapes["ystar"] = (chains, draws, n)
            if not np.isin(y, [0, 1]).all():
                raise ValueError("binary outcomes must be zero or one")
        if self.family in ("poisson", "negative_binomial", "survival"):
            shapes["h"] = (chains, draws, n)
            shapes["latent_white"] = (chains, draws, n)
            if not np.all(self.samples["sigma2"] == 1):
                raise ValueError("generalized models use unit latent variance convention")
        if self.settings.get("clustered", False):
            codes = self.data.get("cluster_codes")
            levels = self.data.get("cluster_levels", [])
            if codes is None or codes.shape != (n,) or not levels:
                raise ValueError("cluster codes and levels are required")
            if not np.issubdtype(codes.dtype, np.integer) or np.any(codes < 0) or np.any(codes >= len(levels)):
                raise ValueError("invalid cluster codes")
            from ._clusters import canonical_label
            if len(set(map(canonical_label, levels))) != len(levels):
                raise ValueError("cluster levels must be distinct typed labels")
            shapes["lambda_cluster"] = (chains, draws)
            if self.family in ("poisson", "negative_binomial", "survival"):
                shapes["u"] = (chains, draws, len(levels))
        if self.family in ("poisson", "negative_binomial"):
            if np.any(y < 0) or np.any(y != np.floor(y)):
                raise ValueError("count outcomes must be nonnegative integers")
            offset = self.data.get("offset")
            if offset is None or offset.shape != (n,) or not np.isfinite(offset).all():
                raise ValueError("invalid count offsets")
        if self.family == "negative_binomial":
            shapes["dispersion"] = (chains, draws)
        if self.family == "survival":
            from ._likelihoods import interval_durations
            breaks = self.data.get("baseline_breaks")
            durations = interval_durations(y, breaks)
            event = self.data.get("event")
            if event is None or event.shape != (n,) or not np.isin(event, [0, 1]).all() or np.any(y <= 0):
                raise ValueError("survival requires positive times and binary event indicators")
            if not np.array_equal(self.data.get("event_bin"), np.searchsorted(breaks, y, side="left")):
                raise ValueError("survival event bins must use right-closed intervals")
            if not np.allclose(self.data.get("durations"), durations, rtol=1e-12, atol=0):
                raise ValueError("survival durations do not match observed times")
            shapes["baseline_hazard"] = (chains, draws, len(breaks) + 1)
        for key, shape in shapes.items():
            if key not in self.samples or self.samples[key].shape != shape:
                raise ValueError(f"invalid shape for posterior {key}")
        for key, value in self.samples.items():
            if not np.issubdtype(value.dtype, np.number) and value.dtype != bool:
                raise ValueError(f"posterior {key} must be numeric")
            if not np.isfinite(value).all():
                raise ValueError(f"posterior {key} contains nonfinite values")
        for key in ("lambda", "sigma2", "lambda_cluster", "dispersion", "baseline_hazard"):
            if key in self.samples and np.any(self.samples[key] <= 0):
                raise ValueError(f"posterior {key} must be positive")
        if np.any(self.samples["r"] < 0) or np.any((self.samples["pi"] < 0) | (self.samples["pi"] > 1)):
            raise ValueError("invalid relevance or inclusion probability")
        if not np.isin(self.samples["delta"], [0, 1]).all():
            raise ValueError("inclusion indicators must be zero or one")
        for key in ("y", "Z", "Z_original", "X", "X_original", "z_center", "z_scale"):
            if not np.isfinite(self.data[key]).all():
                raise ValueError(f"training {key} contains nonfinite values")
        if self.family == "binomial":
            if not np.all(self.samples["sigma2"] == 1):
                raise ValueError("probit residual variance must be one")
            latent = self.samples["ystar"]
            if np.any(np.where(y == 1, latent <= 0, latent >= 0)):
                raise ValueError("latent outcomes must have the observed outcome signs")
        if self.data["groups"] is not None:
            groups = np.asarray(self.data["groups"])
            if groups.shape != (m,):
                raise ValueError("groups must match exposure columns")
            if self.settings.get("selection") == "group":
                for group in np.unique(groups):
                    if np.any(self.samples["delta"][..., groups == group].sum(axis=-1) > 1):
                        raise ValueError("at most one exposure may be selected per group")
        if self.data["X_original"].ndim != 2:
            raise ValueError("original covariates must be a matrix")
        if self.data["Z_original"].shape != z.shape or self.data["X_original"].shape[0] != n:
            raise ValueError("incompatible original training arrays")
        if self.data["z_center"].shape != (m,) or self.data["z_scale"].shape != (m,):
            raise ValueError("invalid exposure transformation")
        if np.any(self.data["z_scale"] <= 0):
            raise ValueError("exposure scales must be positive")
        if len(self.exposure_names) != m or len(self.covariate_names) != p:
            raise ValueError("training names do not match matrix columns")
        if self.family in ("poisson", "negative_binomial", "survival"):
            from ._kernel import gaussian_kernel
            from ._prediction import _check_support
            last_r, basis = None, None
            for chain in range(chains):
                for draw in range(draws):
                    r = self.samples["r"][chain, draw]
                    if last_r is None or not np.array_equal(r, last_r):
                        eig, vectors = np.linalg.eigh(gaussian_kernel(z, r=r))
                        tolerance = 100*n*np.finfo(float).eps*max(1., eig.max())
                        basis = vectors[:, eig > tolerance]
                        last_r = r.copy()
                    # Support is basis invariant. Do not reconstruct h from saved
                    # white coordinates: eigenvector signs/rotations can differ
                    # across valid linear-algebra implementations.
                    _check_support(self.samples["h"][chain, draw], basis)

    @property
    def family(self):
        return self.settings["family"]

    @property
    def exposure_names(self):
        return list(self.data["exposure_names"])

    @property
    def covariate_names(self):
        return list(self.data["covariate_names"])

    def __repr__(self):
        return (f"BKMRFit(family={self.family!r}, n={len(self.data['y'])}, "
                f"exposures={self.data['Z'].shape[1]}, chains={self.settings['chains']}, "
                f"draws={self.settings['draws']})")

    def posterior_samples(self, parameter=None, combine_chains=False):
        """Return saved post-warmup samples, preserving chains by default."""
        if not isinstance(combine_chains, (bool, np.bool_)):
            raise ValueError("combine_chains must be boolean")

        def copy(value):
            return value.reshape((-1,) + value.shape[2:]).copy() if combine_chains else value.copy()

        if parameter is None:
            return {k: copy(v) for k, v in self.samples.items()}
        if parameter not in self.samples:
            raise ValueError(f"unknown parameter: {parameter}")
        return copy(self.samples[parameter])

    def _parameters(self):
        for key in ("lambda", "sigma2", "r", "beta", "pi", "delta", "lambda_cluster", "dispersion", "baseline_hazard", "u"):
            if key not in self.samples or (key == "sigma2" and self.family in ("poisson", "negative_binomial", "survival")):
                continue
            if key in ("pi", "delta") and self.settings.get("selection") == "none":
                continue
            values = self.samples[key]
            if values.ndim == 2:
                yield key, values
            else:
                labels = (self.exposure_names if key in ("r", "delta") else
                          [f"interval{j+1}" for j in range(values.shape[-1])] if key == "baseline_hazard" else
                          [repr(label) for label in self.data["cluster_levels"]] if key == "u" else self.covariate_names)
                for j, label in enumerate(labels):
                    yield f"{key}[{label}]", values[:, :, j]

    def summary_table(self, probability=0.95):
        """Return parameter means, standard deviations and credible interval limits.

        Each interval has equal posterior probability below and above its limits.
        """
        probability = _probability(probability)
        tail = (1 - probability) / 2
        rows = []
        for name, values in self._parameters():
            flat = values.astype(float, copy=False).ravel()
            lower, upper = np.quantile(flat, [tail, 1 - tail])
            rows.append({"parameter": name, "mean": float(flat.mean()),
                         "sd": float(flat.std(ddof=1)), "lower": float(lower),
                         "upper": float(upper)})
        return rows

    def summary(self, probability=0.95):
        """Return a compact text summary without printing or refitting."""
        rows = self.summary_table(probability)
        width = max(12, max(len(row["parameter"]) for row in rows))
        lines = [repr(self), f"{100 * probability:g}% pointwise credible intervals",
                 f"{'Parameter':<{width}} {'Mean':>10} {'SD':>10} {'Lower':>10} {'Upper':>10}"]
        for row in rows:
            lines.append(f"{row['parameter']:<{width}} " + " ".join(
                f"{row[k]:>10.4g}" for k in ("mean", "sd", "lower", "upper")))
        if self.settings.get("selection") != "none":
            lines.append("Posterior inclusion probabilities")
            for row in self.inclusion_probabilities():
                lines.append(f"  {row['variable']}: {row['pip']:.3f}")
        lines.extend(self.sampling_info.get("warnings", []))
        lines.extend(self.diagnostics()["messages"])
        return "\n".join(lines)

    def inclusion_probabilities(self):
        """Return posterior inclusion probabilities (PIPs) for exposure selection.

        An exposure PIP is the proportion of saved states where it is selected.
        Group selection also returns the group's PIP and each exposure's PIP among
        states where its group is selected. This conditional value is undefined
        when the group is never selected.
        """
        if self.settings.get("selection") == "none":
            raise ValueError("inclusion probabilities require a variable-selection fit")
        delta = self.samples["delta"].astype(bool)
        pips = delta.mean(axis=(0, 1))
        rows = []
        groups = self.data["groups"]
        for j, name in enumerate(self.exposure_names):
            row = {"variable": name, "pip": float(pips[j])}
            if groups is not None and self.settings.get("selection") == "group":
                selected = np.asarray(groups) == groups[j]
                group_pip = float(delta[:, :, selected].any(axis=2).mean())
                row.update(group=groups[j], group_pip=group_pip,
                           conditional_pip=float(pips[j] / group_pip) if group_pip else np.nan)
            rows.append(row)
        return rows

    def diagnostics(self):
        """Return R-hat, effective sample sizes and sampling messages.

        R-hat compares chains. Effective sample size accounts for dependence between
        draws, using both the center and tails of their distribution. Favorable
        values do not prove convergence.
        """
        rows, messages = [], []
        for name, values in self._parameters():
            fixed = (name == "sigma2" and self.family == "binomial") or (
                name.startswith("r[") and self.settings.get("r_fixed") is not None)
            if fixed:
                continue
            if np.all(values == values.flat[0]):
                messages.append(f"{name}: constant saved draws; mixing cannot be assessed.")
            rows.append({"parameter": name, **diagnose(values)})
        if self.settings["chains"] < 2:
            messages.append("One chain: between-chain convergence has not been assessed.")
        if self.settings["draws"] < 100:
            messages.append("Few saved draws: diagnostic and tail estimates are imprecise.")
        if any(row["rhat"] > 1.01 for row in rows):
            messages.append("Some R-hat values exceed 1.01; inspect mixing before inference.")
        if any(np.isfinite(row["ess_bulk"]) and row["ess_bulk"] < 100 for row in rows):
            messages.append("Some bulk effective sample sizes are below 100.")
        messages.append("Diagnostics do not establish convergence or validate model assumptions.")
        return {"parameters": rows, "messages": messages,
                "method": "rank-normalized/folded split R-hat; Geyer bulk/tail ESS capped at draw count"}

    def _indices(self, max_draws):
        n = self.settings["draws"]
        if max_draws is None:
            return np.arange(n)
        maximum = _integer(max_draws, "max_draws", 2)
        return np.linspace(0, n - 1, min(n, maximum), dtype=int)

    def _new_exposures(self, Znew):
        if Znew is None:
            return self.data["Z"].copy()
        raw = _matrix(Znew, len(self.exposure_names), self.exposure_names, "Znew")
        return (raw - self.data["z_center"]) / self.data["z_scale"]

    def _new_design(self, Xnew, rows, training):
        k = self.data["X_original"].shape[1]
        intercept = self.settings["intercept"]
        if Xnew is None:
            if training:
                return self.data["X"].copy()
            if k:
                raise ValueError("Xnew is required for new exposures when adjustment covariates were fitted")
            raw = np.empty((rows, 0))
        else:
            names = self.covariate_names[int(intercept):]
            raw = _matrix(Xnew, k, names, "Xnew")
            if len(raw) != rows:
                raise ValueError("Xnew and Znew must have the same number of rows")
        return np.column_stack((np.ones(rows), raw)) if intercept else raw

    def _condition(self, standardized, chain, draw, cluster_new_codes=None):
        from ._kernel import conditional_h

        if self.family in ("poisson", "negative_binomial", "survival"):
            from ._prediction import conditional_saved_h
            return conditional_saved_h(self.data["Z"], standardized,
                                       self.samples["h"][chain, draw],
                                       self.samples["r"][chain, draw],
                                       self.samples["lambda"][chain, draw])
        ycont = self.samples["ystar"][chain, draw] if self.family == "binomial" else self.data["y"]
        return conditional_h(
            self.data["Z"], standardized, ycont, self.data["X"],
            self.samples["beta"][chain, draw], self.samples["r"][chain, draw],
            self.samples["lambda"][chain, draw], self.samples["sigma2"][chain, draw],
            cluster_codes=self.data.get("cluster_codes"),
            lambda_cluster=self.samples["lambda_cluster"][chain, draw] if "lambda_cluster" in self.samples else None,
            cluster_new_codes=cluster_new_codes,
        )

    def _prediction_budget(self, rows, saved, *, moments=False):
        # Account for dense covariance/eigen work, retained output and summaries.
        n = len(self.data["y"])
        output = 0 if moments else 3 * self.settings["chains"] * saved * rows
        estimated = 8 * (output + 8 * rows * rows + 4 * n * rows)
        if estimated > 512 * 1024 * 1024:
            raise ValueError("prediction exceeds the 512 MiB working-memory limit; "
                             "use fewer profiles or max_draws")

    def _h_draws(self, standardized, indices, rng, cluster_new_codes=None):
        if cluster_new_codes is not None:
            return self._clustered_joint_draws(standardized, indices, rng, cluster_new_codes)
        self._prediction_budget(len(standardized), len(indices))
        unique, inverse = np.unique(standardized, axis=0, return_inverse=True)
        result = np.empty((self.settings["chains"], len(indices), len(standardized)))
        for chain in range(self.settings["chains"]):
            for position, draw in enumerate(indices):
                active = self.samples["r"][chain, draw] > 0
                # A selected-out exposure cannot change h. Collapse profiles
                # identical on active dimensions before any floating-point draw.
                if active.any():
                    _, representatives, local_inverse = np.unique(
                        unique[:, active], axis=0, return_index=True, return_inverse=True)
                else:
                    representatives = np.array([0])
                    local_inverse = np.zeros(len(unique), dtype=int)
                mean, covariance = self._condition(unique[representatives], chain, draw)
                local = _normal_draw(mean, covariance, rng)[local_inverse]
                result[chain, position] = local[inverse]
        return result

    def predict_h(self, Znew=None, *, probability=0.95, max_draws=None, seed=None):
        """Draw the exposure function h at profiles supplied in original units.

        Draw all requested rows together to retain their dependence. Exclude linear
        adjustment, random effects and observation noise. Training standardization
        is reused. ``max_draws`` selects evenly spaced saved states per chain;
        it does not refit or extend the chain.
        """
        _probability(probability)
        znew = self._new_exposures(Znew)
        values = self._h_draws(znew, self._indices(max_draws), np.random.default_rng(seed))
        return _prediction(values, probability, "h", "mean")

    def _new_cluster_codes(self, cluster_new, rows):
        if cluster_new is None:
            return None
        if not self.settings.get("clustered", False):
            raise ValueError("cluster_new requires a clustered fit")
        from ._clusters import canonical_label
        values = np.asarray(cluster_new, dtype=object)
        if values.shape != (rows,):
            raise ValueError("cluster_new must have one label per prediction row")
        mapping = {canonical_label(label): j for j, label in enumerate(self.data["cluster_levels"])}
        codes = []
        for label in values:
            key = canonical_label(label)
            if key not in mapping:
                mapping[key] = len(mapping)
            codes.append(mapping[key])
        return np.asarray(codes, dtype=int)

    def _clustered_joint_draws(self, standardized, indices, rng, codes):
        self._prediction_budget(len(standardized), len(indices))
        values = np.empty((self.settings["chains"], len(indices), len(standardized)))
        for chain in range(self.settings["chains"]):
            for position, draw in enumerate(indices):
                active = self.samples["r"][chain, draw] > 0
                keys = np.column_stack((standardized[:, active], codes))
                _, representatives, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
                mean, covariance = self._condition(standardized[representatives], chain, draw,
                                                    cluster_new_codes=codes[representatives])
                values[chain, position] = _normal_draw(mean, covariance, rng)[inverse]
        return values

    def _generalized_u(self, codes, indices, rng):
        values = np.zeros((self.settings["chains"], len(indices), len(codes)))
        known = len(self.data["cluster_levels"])
        unique, inverse = np.unique(codes, return_inverse=True)
        for chain in range(self.settings["chains"]):
            for position, draw in enumerate(indices):
                local = np.empty(len(unique))
                for j, code in enumerate(unique):
                    local[j] = (self.samples["u"][chain, draw, code] if code < known else
                                rng.normal(scale=np.sqrt(self.samples["lambda_cluster"][chain, draw])))
                values[chain, position] = local[inverse]
        return values

    def random_intercepts(self, cluster_new=None, *, probability=.95, max_draws=None, seed=None):
        """Draw one random intercept per training cluster or supplied label.

        Known clusters use their posterior draws. Each unseen label receives a new
        intercept; equal labels share it. Omitted labels use the training-cluster
        order in ``data['cluster_levels']``.
        """
        if not self.settings.get("clustered", False):
            raise ValueError("random_intercepts requires a clustered fit")
        labels = self.data["cluster_levels"] if cluster_new is None else cluster_new
        codes = self._new_cluster_codes(labels, len(labels))
        indices = self._indices(max_draws)
        self._prediction_budget(len(codes), len(indices))
        rng = np.random.default_rng(seed)
        if self.family in ("poisson", "negative_binomial", "survival"):
            values = self._generalized_u(codes, indices, rng)
        else:
            from scipy.linalg import cho_solve

            from ._kernel import covariance_components
            unique, inverse = np.unique(codes, return_inverse=True)
            train = self.data["cluster_codes"]
            cross = (unique[:, None] == train[None, :]).astype(float)
            prior = (unique[:, None] == unique[None, :]).astype(float)
            values = np.empty((self.settings["chains"], len(indices), len(codes)))
            for chain in range(self.settings["chains"]):
                for position, draw in enumerate(indices):
                    amplitude = self.samples["lambda_cluster"][chain, draw]
                    _, _, lower = covariance_components(self.data["Z"], self.samples["r"][chain, draw],
                        self.samples["lambda"][chain, draw], cluster_codes=train, lambda_cluster=amplitude)
                    ycont = self.samples["ystar"][chain, draw] if self.family == "binomial" else self.data["y"]
                    residual = ycont - self.data["X"] @ self.samples["beta"][chain, draw]
                    mean = amplitude * cross @ cho_solve((lower, True), residual)
                    covariance = self.samples["sigma2"][chain, draw] * (
                        amplitude*prior-amplitude**2*cross@cho_solve((lower, True), cross.T))
                    values[chain, position] = _normal_draw(mean, covariance, rng)[inverse]
        return _prediction(values, probability, "random_intercept", "mean")

    def predict(self, Znew=None, Xnew=None, *, kind="mean", scale="response",
                probability=0.95, max_draws=None, seed=None, offset=None, cluster_new=None):
        """Draw outcome means or new observations at requested profiles.

        ``scale='latent'`` returns the predictor before the outcome transformation,
        such as probit or exponentiation. ``scale='response'`` returns outcome
        means, probabilities or relative hazards.
        Omitted ``cluster_new`` sets the random intercept to zero rather than
        averaging over possible clusters. Known labels use their posterior;
        equal unseen labels share one new intercept. Count offsets are log observation
        exposures. Survival means are relative hazards, meaning hazard divided by
        baseline hazard; use ``predict_survival`` for survival or hazard curves.
        """
        if kind not in ("mean", "observation") or scale not in ("response", "latent"):
            raise ValueError("kind must be mean/observation and scale response/latent")
        if self.family == "survival" and kind == "observation":
            raise ValueError("survival observation draws are not supported; use predict_survival")
        count_family = self.family in ("poisson", "negative_binomial")
        if count_family and kind == "observation" and scale == "latent":
            raise ValueError("count observations require scale='response'")
        if offset is not None and not count_family:
            raise ValueError("offset only applies to count predictions")
        _probability(probability)
        znew = self._new_exposures(Znew)
        xnew = self._new_design(Xnew, len(znew), Znew is None)
        codes = self._new_cluster_codes(cluster_new, len(znew))
        indices = self._indices(max_draws)
        rng = np.random.default_rng(seed)
        generalized = self.family in ("poisson", "negative_binomial", "survival")
        values = self._h_draws(znew, indices, rng, None if generalized else codes)
        if generalized and codes is not None:
            values += self._generalized_u(codes, indices, rng)
        values += np.einsum("cdp,np->cdn", self.samples["beta"][:, indices], xnew)
        if count_family:
            if offset is None:
                if Znew is None:
                    offsets = self.data["offset"]
                elif self.settings.get("offset_used", False):
                    raise ValueError("offset is required for new rows because offsets were used in fitting")
                else:
                    offsets = np.zeros(len(znew))
            else:
                offsets = np.asarray(offset, dtype=float)
                if offsets.shape != (len(znew),) or not np.isfinite(offsets).all():
                    raise ValueError("offset must be a finite vector with one entry per prediction row")
            values += offsets
        if self.family == "binomial" and scale == "response":
            values = ndtr(values)
            if kind == "observation":
                values = rng.binomial(1, values).astype(float)
        elif generalized and scale == "response":
            with np.errstate(over="ignore"):
                values = np.exp(values)
            if not np.isfinite(values).all():
                raise FloatingPointError("log-link predictions overflowed; inspect priors and fitted draws")
            if kind == "observation":
                try:
                    if self.family == "poisson":
                        values = rng.poisson(values).astype(float)
                    else:
                        dispersion = self.samples["dispersion"][:, indices, None]
                        values = rng.negative_binomial(dispersion, dispersion/(dispersion+values)).astype(float)
                except ValueError as exc:
                    raise FloatingPointError("predictive counts exceed the random generator's numeric range") from exc
        elif kind == "observation":
            sigma = np.sqrt(self.samples["sigma2"][:, indices, None])
            values += rng.normal(size=values.shape) * sigma
        return _prediction(values, probability, scale, kind)

    def predict_survival(self, Znew=None, Xnew=None, *, times, kind="survival", cluster_new=None,
                         probability=.95, max_draws=None, seed=None):
        """Return survival probabilities or hazards for right-censored survival fits.

        The proportional-hazards model multiplies a shared baseline hazard by each
        profile's relative hazard. A hazard is an event rate among those still at
        risk. Beyond observed follow-up, prediction extends the final constant
        baseline hazard; that assumption needs scientific justification.
        """
        if self.family != "survival":
            raise ValueError("predict_survival requires a survival fit")
        if kind not in ("survival", "hazard"):
            raise ValueError("kind must be survival or hazard")
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite, nonnegative and strictly increasing")
        indices = self._indices(max_draws)
        rows = len(self.data["y"]) if Znew is None else len(np.atleast_2d(Znew))
        estimated = 8 * 6 * self.settings["chains"] * len(indices) * rows * len(times)
        if estimated > 512 * 1024 * 1024:
            raise ValueError("survival prediction exceeds the 512 MiB working-memory limit")
        eta = self.predict(Znew, Xnew, scale="latent", kind="mean", cluster_new=cluster_new,
                           probability=probability, max_draws=max_draws, seed=seed).draws
        baseline = self.samples["baseline_hazard"][:, indices]
        breaks = self.data["baseline_breaks"]
        from ._likelihoods import interval_durations
        cumulative = np.einsum("cdj,tj->cdt", baseline, interval_durations(times, breaks))
        if kind == "survival":
            log_cumulative = np.full_like(cumulative, -np.inf)
            np.log(cumulative, out=log_cumulative, where=cumulative > 0)
            with np.errstate(over="ignore", under="ignore"):
                values = np.exp(-np.exp(eta[..., None] + log_cumulative[:, :, None, :]))
        else:
            selected = baseline[..., np.searchsorted(breaks, times, side="left")]
            with np.errstate(over="ignore"):
                values = np.exp(eta[..., None] + np.log(selected[:, :, None, :]))
            if not np.isfinite(values).all():
                raise FloatingPointError("hazard predictions overflowed")
        return _survival_prediction(values, times, probability, kind)

    def posterior_moments_h(self, Znew=None, max_draws=None):
        """Return the mean and covariance of h averaged over selected saved states.

        At each state, average over the uncertainty in h given the training data.
        Then combine these means and covariances across states, using the state
        count as the denominator for their covariance. R's finite-sample exact
        summary uses one less than that count. Both estimate the same posterior
        moment as sampling error decreases.
        """
        znew = self._new_exposures(Znew)
        indices = self._indices(max_draws)
        self._prediction_budget(len(znew), len(indices), moments=True)
        average = np.zeros(len(znew))
        scatter = np.zeros((len(znew), len(znew)))
        conditional = np.zeros_like(scatter)
        count = 0
        for chain in range(self.settings["chains"]):
            for draw in indices:
                mean, covariance = self._condition(znew, chain, draw)
                count += 1
                difference = mean - average
                average += difference / count
                scatter += np.outer(difference, mean - average)
                conditional += (covariance - conditional) / count
        covariance = conditional + scatter / count
        return {"mean": average, "covariance": (covariance + covariance.T) / 2}

    def save(self, path):
        """Write a compressed, non-pickle archive, replacing the file atomically."""
        path = Path(path)
        arrays = {"data__" + k: v for k, v in self.data.items() if isinstance(v, np.ndarray)}
        arrays.update({"sample__" + k: v for k, v in self.samples.items()})
        metadata = {"schema": 1, "data": {k: v for k, v in self.data.items()
                                         if not isinstance(v, np.ndarray)},
                    "settings": self.settings, "sampling_info": self.sampling_info}

        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"unsupported metadata type: {type(value).__name__}")

        arrays["metadata"] = np.asarray(json.dumps(metadata, default=convert, allow_nan=False))
        descriptor, name = tempfile.mkstemp(prefix=".kernelmix-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                np.savez_compressed(output, **arrays)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path, *, max_bytes=512 * 1024 * 1024):
        """Load a validated non-pickle archive; limit uncompressed size by default."""
        maximum = _integer(max_bytes, "max_bytes")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if sum(item.file_size for item in members) > maximum:
                raise ValueError("fit archive exceeds max_bytes")
            if len({item.filename for item in members}) != len(members):
                raise ValueError("fit archive contains duplicate members")
        with np.load(path, allow_pickle=False) as stored:
            if "metadata" not in stored:
                raise ValueError("fit archive has no metadata")
            metadata = json.loads(str(stored["metadata"].item()))
            if not isinstance(metadata, dict):
                raise ValueError("fit archive metadata must be an object")
            if metadata.get("schema") != 1:
                raise ValueError("unsupported fit archive schema")
            data = metadata["data"]
            samples = {}
            for key in stored.files:
                if key.startswith("data__"):
                    data[key[6:]] = stored[key]
                elif key.startswith("sample__"):
                    samples[key[8:]] = stored[key]
                elif key != "metadata":
                    raise ValueError("unknown fit archive member")
        if not isinstance(metadata.get("settings"), dict) or not isinstance(
                metadata.get("sampling_info"), dict):
            raise ValueError("fit archive settings or sampling information are missing")
        try:
            if metadata["settings"].get("engine") == "structured":
                from .structured_result import StructuredFit
                return StructuredFit(data, samples, metadata["settings"], metadata["sampling_info"])
            return cls(data, samples, metadata["settings"], metadata["sampling_info"])
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError("invalid fit archive structure") from exc
