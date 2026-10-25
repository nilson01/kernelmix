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
    """Joint posterior draws and pointwise equal-tailed intervals.

    ``draws`` has axes (chain, saved draw, prediction row). Covariance includes
    posterior variation and the conditional GP uncertainty. Mean predictions
    exclude observation noise; observation predictions include it.
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
        if self.family not in ("gaussian", "binomial"):
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
        for key, shape in shapes.items():
            if key not in self.samples or self.samples[key].shape != shape:
                raise ValueError(f"invalid shape for posterior {key}")
        for key, value in self.samples.items():
            if not np.issubdtype(value.dtype, np.number) and value.dtype != bool:
                raise ValueError(f"posterior {key} must be numeric")
            if not np.isfinite(value).all():
                raise ValueError(f"posterior {key} contains nonfinite values")
        for key in ("lambda", "sigma2"):
            if np.any(self.samples[key] <= 0):
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
        for key in ("lambda", "sigma2", "r", "beta", "pi", "delta"):
            if key in ("pi", "delta") and self.settings.get("selection") == "none":
                continue
            values = self.samples[key]
            if values.ndim == 2:
                yield key, values
            else:
                labels = self.exposure_names if key in ("r", "delta") else self.covariate_names
                for j, label in enumerate(labels):
                    yield f"{key}[{label}]", values[:, :, j]

    def summary_table(self, probability=0.95):
        """Posterior mean, SD and pointwise equal-tailed credible limits."""
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
        lines.extend(self.diagnostics()["messages"])
        return "\n".join(lines)

    def inclusion_probabilities(self):
        """Return component PIPs, and group/conditional PIPs for grouped selection."""
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
        """Return rank/folded split R-hat, bulk/tail ESS, and diagnostic messages."""
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

    def _condition(self, standardized, chain, draw):
        from ._kernel import conditional_h

        ycont = self.samples["ystar"][chain, draw] if self.family == "binomial" else self.data["y"]
        return conditional_h(
            self.data["Z"], standardized, ycont, self.data["X"],
            self.samples["beta"][chain, draw], self.samples["r"][chain, draw],
            self.samples["lambda"][chain, draw], self.samples["sigma2"][chain, draw],
        )

    def _prediction_budget(self, rows, saved, *, moments=False):
        # Account for dense covariance/eigen work, retained output and summaries.
        n = len(self.data["y"])
        output = 0 if moments else 3 * self.settings["chains"] * saved * rows
        estimated = 8 * (output + 8 * rows * rows + 4 * n * rows)
        if estimated > 512 * 1024 * 1024:
            raise ValueError("prediction exceeds the 512 MiB working-memory limit; "
                             "use fewer profiles or max_draws")

    def _h_draws(self, standardized, indices, rng):
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
        """Sample the joint GP exposure function on original exposure units.

        Training standardization is reused. ``max_draws`` limits saved states per
        chain by evenly spaced selection; it does not refit or extend the chain.
        """
        _probability(probability)
        znew = self._new_exposures(Znew)
        values = self._h_draws(znew, self._indices(max_draws), np.random.default_rng(seed))
        return _prediction(values, probability, "h", "mean")

    def predict(self, Znew=None, Xnew=None, *, kind="mean", scale="response",
                probability=0.95, max_draws=None, seed=None):
        """Sample joint outcome means or new observations.

        Binary response means are probabilities under the probit link. Gaussian
        response means are on the outcome scale. New-data covariates exclude the
        intercept column. Credible/predictive limits are posterior quantiles.
        """
        if kind not in ("mean", "observation") or scale not in ("response", "latent"):
            raise ValueError("kind must be mean/observation and scale response/latent")
        _probability(probability)
        znew = self._new_exposures(Znew)
        xnew = self._new_design(Xnew, len(znew), Znew is None)
        indices = self._indices(max_draws)
        rng = np.random.default_rng(seed)
        values = self._h_draws(znew, indices, rng)
        values += np.einsum("cdp,np->cdn", self.samples["beta"][:, indices], xnew)
        if self.family == "binomial" and scale == "response":
            values = ndtr(values)
            if kind == "observation":
                values = rng.binomial(1, values).astype(float)
        elif kind == "observation":
            sigma = np.sqrt(self.samples["sigma2"][:, indices, None])
            values += rng.normal(size=values.shape) * sigma
        return _prediction(values, probability, scale, kind)

    def posterior_moments_h(self, Znew=None, max_draws=None):
        """Integrate conditional GP moments over selected saved states.

        Uses the empirical-distribution (population) covariance of conditional
        means. R's finite-sample exact-summary method uses sample covariance;
        both converge to the same posterior moment as sampling error decreases.
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
            return cls(data, samples, metadata["settings"], metadata["sampling_info"])
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError("invalid fit archive structure") from exc
