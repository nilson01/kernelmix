"""Portable results for structured models and finite GP representations."""

from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from ._basis import _eigen, _supported_values, basis_matrix, basis_predict, kernel_matrix
from ._clusters import canonical_label
from ._structure import correlation_blocks, residual_prediction
from ._survival import prepare_extended_survival, survival_curves, survival_path_curves
from .result import (
    BKMRFit,
    _freeze,
    _integer,
    _normal_draw,
    _prediction,
    _probability,
    _survival_prediction,
)


@dataclass(frozen=True)
class RandomCoefficientPrediction:
    """Random-intercept and slope draws with summaries for each cluster.

    ``draws`` has axes (chain, saved draw, cluster label, coefficient).
    Intervals apply to each coefficient separately.
    """

    draws: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    labels: tuple
    coefficients: tuple
    probability: float

    def to_records(self):
        return [
            dict(
                cluster=label,
                coefficient=name,
                mean=float(self.mean[i, j]),
                sd=float(self.sd[i, j]),
                lower=float(self.lower[i, j]),
                upper=float(self.upper[i, j]),
            )
            for i, label in enumerate(self.labels)
            for j, name in enumerate(self.coefficients)
        ]


@dataclass(frozen=True)
class CompetingRiskPrediction:
    """Probabilities of each cause occurring by each requested time.

    These cumulative incidences account for the other competing causes.
    ``draws`` has axes (chain, saved draw, row, time, cause). Intervals apply to
    each row/time/cause separately.
    """

    draws: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    times: np.ndarray
    probability: float
    causes: tuple
    kind: str = "cumulative_incidence"

    def for_cause(self, cause):
        index = _integer(cause, "cause") - 1
        if index >= len(self.causes):
            raise ValueError("cause exceeds the fitted number of causes")
        return _survival_prediction(self.draws[..., index], self.times, self.probability, self.kind)

    def to_records(self):
        return [
            dict(
                row=i,
                time=float(t),
                cause=cause,
                mean=float(self.mean[i, j, k]),
                sd=float(self.sd[i, j, k]),
                lower=float(self.lower[i, j, k]),
                upper=float(self.upper[i, j, k]),
            )
            for i in range(self.mean.shape[0])
            for j, t in enumerate(self.times)
            for k, cause in enumerate(self.causes)
        ]


class StructuredFit(BKMRFit):
    """Posterior results for kernel choices, GP bases and structured outcomes.

    A kernel defines similarity between exposure profiles. A Gaussian process
    (GP) uses it to model the exposure function h. Stored ``lambda`` is the
    direct GP variance tau from Bobb et al. (2015), including for Gaussian
    outcomes. The default Gaussian/probit sampler instead stores tau/sigma2.
    Omitting prediction cluster labels sets random effects to zero. With delayed
    entry, latent predictors have priors for the included entrant cohort; this
    does not recover the population distribution before entry.
    """

    def _basis(self):
        result = deepcopy(self.data["basis_meta"])
        result.update(
            {
                name[6:]: value
                for name, value in self.data.items()
                if name.startswith("basis_") and name != "basis_meta"
            }
        )
        return result

    def _validate(self):
        from ._structured import _rank

        required = {
            "y",
            "Z",
            "Z_original",
            "X",
            "X_original",
            "z_center",
            "z_scale",
            "rows",
            "original_n",
            "exposure_names",
            "covariate_names",
            "groups",
            "basis_meta",
            "random_design",
            "variance_design",
            "cluster_codes",
            "cluster_levels",
        }
        if not required.issubset(self.data):
            raise ValueError("structured fit data are incomplete")
        if (
            self.settings.get("engine") != "structured"
            or self.settings.get("amplitude_convention") != "direct_tau"
        ):
            raise ValueError("structured model metadata are inconsistent")
        if self.family not in ("gaussian", "binomial", "poisson", "negative_binomial", "survival"):
            raise ValueError("unknown structured family")
        n, m = self.data["Z"].shape
        p = self.data["X"].shape[1]
        c = _integer(self.settings["chains"], "chains")
        d = _integer(self.settings["draws"], "draws", 2)
        k = _integer(self.settings["n_causes"], "n_causes")
        g = len(self.data["cluster_levels"])
        q = self.data["random_design"].shape[1]
        v = self.data["variance_design"].shape[1]
        if self.family != "survival" and k != 1:
            raise ValueError("only survival fits have multiple causes")
        if n < 2 or not m or self.data["y"].shape != (n,) or self.data["X"].shape != (n, p):
            raise ValueError("invalid training data shapes")
        for name in (
            "y",
            "Z",
            "Z_original",
            "X",
            "X_original",
            "z_center",
            "z_scale",
            "random_design",
            "variance_design",
        ):
            if not np.isfinite(self.data[name]).all():
                raise ValueError(f"nonfinite fit data: {name}")
        if (
            self.data["Z_original"].shape != (n, m)
            or self.data["z_center"].shape != (m,)
            or self.data["z_scale"].shape != (m,)
            or (self.data["z_scale"] <= 0).any()
        ):
            raise ValueError("invalid exposure transformation")
        if not np.allclose(
            self.data["Z"],
            (self.data["Z_original"] - self.data["z_center"]) / self.data["z_scale"],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("inconsistent transformed exposures")
        if self.data["X_original"].shape != (n, p - int(self.settings["intercept"])):
            raise ValueError("invalid raw adjustment shape")
        names = [self.exposure_names, self.covariate_names]
        if (
            any(
                len(set(a)) != len(a) or any(not isinstance(s, str) or not s for s in a)
                for a in names
            )
            or len(names[0]) != m
            or len(names[1]) != p
        ):
            raise ValueError("invalid variable names")
        expected_x = (
            np.column_stack((np.ones(n), self.data["X_original"]))
            if self.settings["intercept"]
            else self.data["X_original"]
        )
        if not np.array_equal(self.data["X"], expected_x):
            raise ValueError("invalid adjustment transformation")
        rows = self.data["rows"]
        original_n = _integer(self.data["original_n"], "original_n", 2)
        if (
            rows.shape != (n,)
            or not np.issubdtype(rows.dtype, np.integer)
            or (rows < 0).any()
            or (rows >= original_n).any()
            or (np.diff(rows) <= 0).any()
        ):
            raise ValueError("invalid retained original rows")
        if self.data["random_design"].shape != (n, q) or self.data["variance_design"].shape != (
            n,
            v,
        ):
            raise ValueError("invalid extra design rows")
        _rank(self.data["random_design"], "random coefficient design")
        _rank(self.data["variance_design"], "variance_design", no_constant=True)
        if g:
            codes = self.data["cluster_codes"]
            if (
                codes.shape != (n,)
                or not np.issubdtype(codes.dtype, np.integer)
                or (codes < 0).any()
                or (codes >= g).any()
                or q < 1
                or not np.all(self.data["random_design"][:, 0] == 1)
            ):
                raise ValueError("invalid cluster design")
            if len(set(map(canonical_label, self.data["cluster_levels"]))) != g:
                raise ValueError("cluster levels must be distinct")
        elif q or self.data["cluster_codes"] is not None:
            raise ValueError("unclustered fits cannot have a random design")
        basis = self._basis()
        b = _integer(basis["n_features"], "n_features")
        if (
            basis["n_training"] != n
            or basis["n_exposures"] != m
            or any(basis[key] != self.settings[key] for key in ("kernel", "approximation"))
            or basis["alpha"] != self.settings["kernel_alpha"]
        ):
            raise ValueError("basis metadata do not match the fitted model")
        mode = basis["approximation"]
        if mode == "dense":
            if (
                b != n
                or np.shape(basis.get("train_Z")) != (n, m)
                or not np.array_equal(basis["train_Z"], self.data["Z"])
            ):
                raise ValueError("dense basis anchors must match retained transformed exposures")
        elif mode == "pp":
            if np.shape(basis.get("knots")) != (b, m):
                raise ValueError("predictive-process knots must match n_features and exposures")
            knot_rows = np.asarray(basis.get("knot_indices", []))
            if basis.get("knot_selection") == "maxmin":
                if (
                    knot_rows.shape != (b,)
                    or not np.issubdtype(knot_rows.dtype, np.integer)
                    or (knot_rows < 0).any()
                    or (knot_rows >= n).any()
                    or not np.array_equal(basis["knots"], self.data["Z"][knot_rows])
                ):
                    raise ValueError("automatic knots and retained training rows disagree")
        elif mode == "rff":
            pairs = _integer(basis["n_components"], "n_components")
            if (
                b != 2 * pairs
                or np.shape(basis.get("base_frequencies")) != (pairs, m)
                or np.shape(basis.get("frequency_scales")) != (pairs,)
            ):
                raise ValueError("frozen Fourier arrays and feature counts disagree")
        else:
            raise ValueError("unknown saved approximation")
        basis_matrix(self.data["Z"][:1], np.ones(m), basis)
        shapes = {
            "beta": (k, p),
            "r": (m,),
            "lambda": (k,),
            "sigma2": (),
            "delta": (m,),
            "pi": (),
            "gp_weight": (k, b),
        }
        if basis["approximation"] != "rff":
            shapes["gp_anchor"] = (
                k,
                n if basis["approximation"] == "dense" else len(basis["knots"]),
            )
        if g:
            shapes.update(random_coeff=(k, g, q), random_covariance=(k, q, q))
        if v:
            shapes["gamma"] = (v,)
        corr = self.settings["correlation"]
        if corr in ("ar1", "exponential"):
            shapes["correlation_parameter"] = ()
        if self.family == "negative_binomial":
            shapes["dispersion"] = ()
        if self.family == "survival":
            shapes["baseline_hazard"] = (k, len(self.data["baseline_breaks"]) + 1)
        if set(self.samples) != set(shapes):
            raise ValueError("structured sample fields do not match the model")
        for name, shape in shapes.items():
            values = self.samples[name]
            if values.shape != (c, d) + shape or not np.isfinite(values).all():
                raise ValueError(f"invalid saved parameter: {name}")
        for name in ("lambda", "sigma2", "dispersion", "baseline_hazard"):
            if name in self.samples and (self.samples[name] <= 0).any():
                raise ValueError(f"{name} must be positive")
        if self.family != "gaussian" and not np.all(self.samples["sigma2"] == 1):
            raise ValueError("non-Gaussian residual convention is one")
        r, delta = self.samples["r"], self.samples["delta"]
        if (
            (r < 0).any()
            or not np.isin(delta, [0, 1]).all()
            or np.any((r > 0) != delta.astype(bool))
        ):
            raise ValueError("saved relevance and indicators disagree")
        selection = self.settings["selection"]
        if selection not in ("none", "component", "group"):
            raise ValueError("invalid selection mode")
        pi = self.samples["pi"]
        if (selection == "none" and not np.all(pi == 1)) or (
            selection != "none" and not ((pi > 0) & (pi < 1)).all()
        ):
            raise ValueError("invalid inclusion probabilities")
        fixed = self.settings.get("r_fixed")
        if fixed is not None and (selection != "none" or not np.all(r == np.asarray(fixed))):
            raise ValueError("saved fixed relevance is inconsistent")
        if selection == "none" and fixed is None and not delta.all():
            raise ValueError("without selection all relevance values must be active")
        if selection == "group":
            groups = np.asarray(self.data["groups"])
            if groups.shape != (m,) or any(
                (delta[..., groups == label].sum(axis=-1) > 1).any() for label in set(groups)
            ):
                raise ValueError("invalid grouped selection states")
        if g:
            covariance = self.samples["random_covariance"]
            if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-12):
                raise ValueError("random covariance must be symmetric")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("random covariance must be positive definite") from exc
            if self.settings["random_covariance_fixed"] and not np.allclose(
                covariance, self.data["fixed_random_covariance"], rtol=1e-12, atol=1e-12
            ):
                raise ValueError("fixed random covariance changed")
        if self.family == "binomial" and not np.isin(self.data["y"], [0, 1]).all():
            raise ValueError("invalid binary outcomes")
        if self.family in ("poisson", "negative_binomial"):
            if (
                (self.data["y"] < 0).any()
                or (self.data["y"] != np.floor(self.data["y"])).any()
                or self.data["offset"].shape != (n,)
                or not np.isfinite(self.data["offset"]).all()
            ):
                raise ValueError("invalid counts or offsets")
        if self.family == "survival":
            labels = [self.data["subject_levels"][int(i)] for i in self.data["subject_codes"]]
            checked = prepare_extended_survival(
                self.data["y"],
                self.data["event"],
                entry=self.data["entry"],
                upper=self.data["upper"],
                subject=labels,
                n_causes=k,
                baseline_breaks=self.data["baseline_breaks"],
            )
            for name in (
                "durations",
                "interval_durations",
                "event_bin",
                "is_interval",
                "terminal",
                "subject_codes",
            ):
                if not np.array_equal(self.data[name], checked[name]):
                    raise ValueError(f"inconsistent survival data: {name}")
        if corr != "independent":
            if self.family != "gaussian" or not g:
                raise ValueError("residual correlation requires clustered Gaussian data")
            for value in (
                np.unique(self.samples["correlation_parameter"])
                if "correlation_parameter" in self.samples
                else [None]
            ):
                correlation_blocks(
                    self.data["cluster_codes"],
                    self.data["observation_time"],
                    corr,
                    value,
                    self.data["correlation_matrix"],
                )
        if "gp_anchor" in self.samples:
            anchors = basis["train_Z"] if basis["approximation"] == "dense" else basis["knots"]
            last = None
            for chain in range(c):
                for draw in range(d):
                    relevance = r[chain, draw]
                    if last is None or not np.array_equal(last, relevance):
                        eigen, vectors = _eigen(
                            kernel_matrix(
                                anchors, r=relevance, kernel=basis["kernel"], alpha=basis["alpha"]
                            )
                        )
                        last = relevance.copy()
                    for cause in range(k):
                        _supported_values(
                            self.samples["gp_anchor"][chain, draw, cause],
                            eigen,
                            vectors,
                            "gp_anchor",
                        )
        self._cause_index()

    def _cause_index(self):
        cause = _integer(self.settings.get("active_cause", 1), "cause")
        if cause > self.settings["n_causes"]:
            raise ValueError("cause exceeds the fitted number of causes")
        return cause - 1

    def for_cause(self, cause):
        """Return a view of h, relative hazards and contrasts for one cause.

        Causes are numbered from one. The view shares stored data and samples.
        """
        if self.family != "survival":
            raise ValueError("for_cause requires a survival fit")
        result = copy(self)
        result.settings = deepcopy(self.settings)
        result.settings["active_cause"] = cause
        result._cause_index()
        return result

    def _parameters(self):
        for key in (
            "lambda",
            "sigma2",
            "r",
            "beta",
            "pi",
            "delta",
            "random_covariance",
            "random_coeff",
            "gamma",
            "correlation_parameter",
            "dispersion",
            "baseline_hazard",
        ):
            if (
                key not in self.samples
                or (key == "sigma2" and self.family != "gaussian")
                or (key in ("pi", "delta") and self.settings["selection"] == "none")
            ):
                continue
            values = self.samples[key]
            if values.ndim == 2:
                yield key, values
                continue
            for index in np.ndindex(values.shape[2:]):
                if key in ("r", "delta"):
                    label = self.exposure_names[index[0]]
                elif key == "beta":
                    label = f"cause{index[0] + 1},{self.covariate_names[index[1]]}"
                elif key == "baseline_hazard":
                    label = f"cause{index[0] + 1},interval{index[1] + 1}"
                elif key == "random_coeff":
                    label = f"cause{index[0] + 1},{self.data['cluster_levels'][index[1]]!r},coefficient{index[2]}"
                else:
                    label = ",".join(str(i + 1) for i in index)
                yield f"{key}[{label}]", values[(slice(None), slice(None)) + index]

    def _prediction_budget(self, rows, saved, *, moments=False):
        b = self.data["basis_meta"]["n_features"]
        n = len(self.data["y"]) if self.settings["approximation"] == "dense" else b
        elements = (
            8 * rows * rows
            + 4 * n * rows
            + (0 if moments else 3 * self.settings["chains"] * saved * rows)
        )
        if 8 * elements > 512 * 1024**2:
            raise ValueError(
                "prediction exceeds the 512 MiB memory limit; use smaller batches or max_draws"
            )

    def _condition(self, znew, chain, draw, cluster_new_codes=None):
        if cluster_new_codes is not None:
            raise ValueError("structured random effects are added separately")
        cause = self._cause_index()
        basis = self._basis()
        r = self.samples["r"][chain, draw]
        amplitude = self.samples["lambda"][chain, draw, cause]
        if basis["approximation"] == "rff":
            return np.sqrt(amplitude) * (
                basis_matrix(znew, r, basis) @ self.samples["gp_weight"][chain, draw, cause]
            ), np.zeros((len(znew), len(znew)))
        anchors = basis["train_Z"] if basis["approximation"] == "dense" else basis["knots"]
        values, vectors = _eigen(
            kernel_matrix(anchors, r=r, kernel=basis["kernel"], alpha=basis["alpha"])
        )
        h = _supported_values(
            self.samples["gp_anchor"][chain, draw, cause], values, vectors, "gp_anchor"
        )
        inverse = np.zeros_like(values)
        np.divide(1.0, values, out=inverse, where=values > 0)
        cross = kernel_matrix(znew, anchors, r, kernel=basis["kernel"], alpha=basis["alpha"])
        pinv = (vectors * inverse) @ vectors.T
        mean = cross @ (pinv @ h)
        covariance = np.zeros((len(znew), len(znew)))
        if basis["approximation"] == "dense":
            covariance = amplitude * (
                kernel_matrix(znew, r=r, kernel=basis["kernel"], alpha=basis["alpha"])
                - cross @ pinv @ cross.T
            )
            covariance = (covariance + covariance.T) / 2
        active = r > 0
        for j, row in enumerate(znew):
            matches = np.flatnonzero(np.all(anchors[:, active] == row[active], axis=1))
            if matches.size:
                mean[j] = h[matches[0]]
                covariance[j, :] = 0.0
                covariance[:, j] = 0.0
        return mean, covariance

    def _coefficient_draws(self, labels, indices, rng):
        codes = self._new_cluster_codes(labels, len(labels))
        cause = self._cause_index()
        q = self.data["random_design"].shape[1]
        g = len(self.data["cluster_levels"])
        values = np.empty((self.settings["chains"], len(indices), len(codes), q))
        unique, inverse = np.unique(codes, return_inverse=True)
        for chain in range(self.settings["chains"]):
            for position, draw in enumerate(indices):
                local = np.empty((len(unique), q))
                for j, code in enumerate(unique):
                    local[j] = (
                        self.samples["random_coeff"][chain, draw, cause, code]
                        if code < g
                        else _normal_draw(
                            np.zeros(q), self.samples["random_covariance"][chain, draw, cause], rng
                        )
                    )
                values[chain, position] = local[inverse]
        return values

    def random_effects(self, cluster_new=None, *, probability=0.95, max_draws=None, seed=None):
        """Draw random intercept/slope coefficients for supplied or training labels.

        Known labels use their saved coefficients; equal unseen labels share one new
        coefficient vector per draw. Omitted labels use training-cluster order.
        """
        if not self.settings["clustered"]:
            raise ValueError("random_effects requires a clustered fit")
        _probability(probability)
        labels = self.data["cluster_levels"] if cluster_new is None else list(cluster_new)
        indices = self._indices(max_draws)
        self._prediction_budget(len(labels), len(indices))
        values = self._coefficient_draws(labels, indices, np.random.default_rng(seed))
        flat = values.reshape((-1,) + values.shape[2:])
        tail = (1 - probability) / 2
        low, high = np.quantile(flat, [tail, 1 - tail], axis=0)
        return RandomCoefficientPrediction(
            *map(_freeze, (values, flat.mean(axis=0), flat.std(axis=0, ddof=1), low, high)),
            tuple(labels),
            tuple(["Intercept"] + [f"Slope{j}" for j in range(1, values.shape[-1])]),
            probability,
        )

    def random_intercepts(self, cluster_new=None, *, probability=0.95, max_draws=None, seed=None):
        result = self.random_effects(
            cluster_new, probability=probability, max_draws=max_draws, seed=seed
        )
        return _prediction(result.draws[..., 0], probability, "random_intercept", "mean")

    def _new_random_design(self, slopes, rows, training):
        q = self.data["random_design"].shape[1]
        if not q:
            if slopes is not None:
                raise ValueError("random_slopes_new requires a random-slope fit")
            return np.empty((rows, 0))
        if slopes is None:
            if training:
                return self.data["random_design"]
            if q > 1:
                raise ValueError(
                    "random_slopes_new is required when predicting random-slope contributions"
                )
            return np.ones((rows, 1))
        raw = np.asarray(slopes, dtype=float)
        if raw.ndim == 1:
            raw = raw[:, None]
        if raw.shape != (rows, q - 1) or not np.isfinite(raw).all():
            raise ValueError("random_slopes_new must match the fitted slope columns")
        return np.column_stack((np.ones(rows), raw))

    def _new_variance_design(self, design, rows, training, needed):
        v = self.data["variance_design"].shape[1]
        if design is None:
            if training:
                return self.data["variance_design"]
            if v and needed:
                raise ValueError(
                    "variance_design_new is required for residual variances or correlated observations"
                )
            return np.zeros((rows, v))
        raw = np.asarray(design, dtype=float)
        if raw.ndim == 1:
            raw = raw[:, None]
        if raw.shape != (rows, v) or not np.isfinite(raw).all():
            raise ValueError("variance_design_new must match the fitted variance columns")
        return raw

    def predict_residual_variance(
        self, Znew=None, *, variance_design_new=None, probability=0.95, max_draws=None
    ):
        """Return residual variance draws, sigma2*exp(V gamma), for Gaussian outcomes.

        V is the variance design and gamma its coefficients. Residual variation is
        outcome variation left after accounting for h, adjustment and random effects.
        These draws exclude uncertainty in h and the random effects.
        """
        if self.family != "gaussian":
            raise ValueError("residual variance prediction requires Gaussian outcomes")
        rows = len(self._new_exposures(Znew))
        indices = self._indices(max_draws)
        self._prediction_budget(rows, len(indices))
        v = self._new_variance_design(variance_design_new, rows, Znew is None, True)
        values = np.broadcast_to(
            self.samples["sigma2"][:, indices, None], (self.settings["chains"], len(indices), rows)
        ).copy()
        if v.shape[1]:
            values *= np.exp(np.einsum("cdv,nv->cdn", self.samples["gamma"][:, indices], v))
        return _prediction(values, probability, "residual_variance", "mean")

    def predict(
        self,
        Znew=None,
        Xnew=None,
        *,
        kind="mean",
        scale="response",
        probability=0.95,
        max_draws=None,
        seed=None,
        offset=None,
        cluster_new=None,
        random_slopes_new=None,
        variance_design_new=None,
        observation_time_new=None,
        correlation_cross=None,
        correlation_new=None,
    ):
        """Draw outcome means or observations with cluster and residual inputs.

        Omitted cluster labels set random effects to zero. Known labels use saved
        coefficients; equal unseen labels share new coefficients. ``latent`` returns
        the predictor before the outcome transformation; ``response`` returns means,
        probabilities or relative hazards. For correlated Gaussian residuals, known
        cluster labels also allow prediction to use the recorded residual process;
        observation draws include the remaining residual uncertainty.
        """
        if kind not in ("mean", "observation") or scale not in ("response", "latent"):
            raise ValueError("invalid prediction kind or scale")
        if self.family != "gaussian" and any(
            a is not None
            for a in (variance_design_new, observation_time_new, correlation_cross, correlation_new)
        ):
            raise ValueError("residual variance/correlation inputs require Gaussian predictions")
        if self.family == "survival" and kind == "observation":
            raise ValueError("use survival probabilities instead of survival observation draws")
        if (
            self.family in ("poisson", "negative_binomial")
            and kind == "observation"
            and scale == "latent"
        ):
            raise ValueError("count observations require response scale")
        if offset is not None and self.family not in ("poisson", "negative_binomial"):
            raise ValueError("offset applies only to counts")
        _probability(probability)
        z = self._new_exposures(Znew)
        rows = len(z)
        indices = self._indices(max_draws)
        x = self._new_design(Xnew, rows, Znew is None)
        rng = np.random.default_rng(seed)
        cause = self._cause_index()
        codes = self._new_cluster_codes(cluster_new, rows)
        values = self._h_draws(z, indices, rng)
        values += np.einsum("cdp,np->cdn", self.samples["beta"][:, indices, cause], x)
        if codes is not None:
            w = self._new_random_design(random_slopes_new, rows, Znew is None)
            coefficients = self._coefficient_draws(cluster_new, indices, rng)
            values += np.einsum("cdnq,nq->cdn", coefficients, w)
        elif random_slopes_new is not None:
            self._new_random_design(random_slopes_new, rows, Znew is None)
        if self.family in ("poisson", "negative_binomial"):
            offsets = (
                self.data["offset"]
                if offset is None and Znew is None
                else np.zeros(rows)
                if offset is None and not self.settings["offset_used"]
                else np.asarray(offset, dtype=float)
            )
            if offsets.shape != (rows,) or not np.isfinite(offsets).all():
                raise ValueError("supply a finite log offset per new row when offsets were fitted")
            values += offsets
        if self.family == "gaussian" and (
            kind == "observation"
            or self.settings["correlation"] != "independent"
            and codes is not None
        ):
            correlated = self.settings["correlation"] != "independent"
            needs_variance = kind == "observation" or (correlated and codes is not None)
            v = self._new_variance_design(variance_design_new, rows, Znew is None, needs_variance)
            time_new = observation_time_new
            if correlated and self.settings["correlation"] in ("ar1", "exponential"):
                if time_new is None:
                    if Znew is None:
                        time_new = self.data["observation_time"]
                    elif codes is None:
                        time_new = np.zeros(rows)
                    else:
                        raise ValueError("observation_time_new is required for correlated new rows")
            newcodes = (
                codes if codes is not None else len(self.data["cluster_levels"]) + np.arange(rows)
            )
            for chain in range(self.settings["chains"]):
                for position, draw in enumerate(indices):
                    gamma = (
                        self.samples["gamma"][chain, draw]
                        if "gamma" in self.samples
                        else np.empty(0)
                    )
                    train_eta = self._training_eta(chain, draw)[:, 0]
                    cross, newmatrix = correlation_cross, correlation_new
                    if (
                        correlated
                        and self.settings["correlation"] == "fixed"
                        and Znew is None
                        and codes is not None
                        and np.array_equal(codes, self.data["cluster_codes"])
                        and cross is None
                        and newmatrix is None
                    ):
                        cross = newmatrix = self.data["correlation_matrix"]
                    mean, covariance = residual_prediction(
                        self.data["y"],
                        train_eta,
                        self.samples["sigma2"][chain, draw],
                        self.data["variance_design"] @ gamma,
                        v @ gamma,
                        self.data["cluster_codes"],
                        newcodes,
                        self.data["observation_time"],
                        time_new,
                        self.settings["correlation"],
                        self.samples["correlation_parameter"][chain, draw]
                        if "correlation_parameter" in self.samples
                        else None,
                        rng,
                        matrix=self.data["correlation_matrix"],
                        cross=cross,
                        new_matrix=newmatrix,
                        observation=kind == "observation",
                    )
                    values[chain, position] += (
                        _normal_draw(mean, covariance, rng) if kind == "observation" else mean
                    )
        elif self.family == "binomial":
            if scale == "response":
                values = ndtr(values)
                if kind == "observation":
                    values = rng.binomial(1, values).astype(float)
            elif kind == "observation":
                values += rng.normal(size=values.shape)
        elif self.family in ("poisson", "negative_binomial", "survival") and scale == "response":
            with np.errstate(over="ignore"):
                values = np.exp(values)
            if not np.isfinite(values).all():
                raise FloatingPointError("log-link predictions overflowed")
            if kind == "observation":
                try:
                    if self.family == "poisson":
                        values = rng.poisson(values).astype(float)
                    else:
                        phi = self.samples["dispersion"][:, indices, None]
                        values = rng.negative_binomial(phi, phi / (phi + values)).astype(float)
                except ValueError as exc:
                    raise FloatingPointError("predictive counts exceed generator limits") from exc
        return _prediction(values, probability, scale, kind)

    def _training_eta(self, chain, draw):
        basis = self._basis()
        k = self.settings["n_causes"]
        eta = np.empty((len(self.data["y"]), k))
        if basis["approximation"] == "dense":
            h = self.samples["gp_anchor"][chain, draw].T
        else:
            h = np.column_stack(
                [
                    basis_predict(
                        self.data["Z"],
                        self.samples["r"][chain, draw],
                        basis,
                        self.samples["gp_weight"][chain, draw, j],
                        self.samples["lambda"][chain, draw, j],
                        None,
                        training_h=self.samples["gp_anchor"][chain, draw, j]
                        if "gp_anchor" in self.samples
                        else None,
                    )
                    for j in range(k)
                ]
            )
        eta[:] = h + self.data["X"] @ self.samples["beta"][chain, draw].T
        if self.settings["clustered"]:
            eta += np.einsum(
                "nkq,nq->nk",
                self.samples["random_coeff"][chain, draw][
                    :, self.data["cluster_codes"], :
                ].transpose(1, 0, 2),
                self.data["random_design"],
            )
        return eta

    def _all_eta(self, Znew, Xnew, *, indices, seed, cluster_new, random_slopes_new):
        rng = np.random.default_rng(seed)
        return np.stack(
            [
                self.for_cause(j + 1)
                .predict(
                    Znew,
                    Xnew,
                    scale="latent",
                    max_draws=len(indices),
                    seed=rng,
                    cluster_new=cluster_new,
                    random_slopes_new=random_slopes_new,
                )
                .draws
                for j in range(self.settings["n_causes"])
            ],
            axis=-1,
        )

    def _survival_draws(
        self,
        Znew,
        Xnew,
        *,
        times,
        entry,
        cluster_new,
        random_slopes_new,
        probability,
        max_draws,
        seed,
        path=None,
    ):
        if self.family != "survival":
            raise ValueError("survival curves require a survival fit")
        _probability(probability)
        indices = self._indices(max_draws)
        rows = len(self._new_exposures(Znew))
        t = np.asarray(times, dtype=float)
        if (
            8
            * 6
            * self.settings["chains"]
            * len(indices)
            * rows
            * t.size
            * (self.settings["n_causes"] + 1)
            > 512 * 1024**2
        ):
            raise ValueError("survival output exceeds the 512 MiB memory limit")
        eta = self._all_eta(
            Znew,
            Xnew,
            indices=indices,
            seed=seed,
            cluster_new=cluster_new,
            random_slopes_new=random_slopes_new,
        )
        collected = []
        incidences = []
        hazards = []
        levels = None
        for chain in range(self.settings["chains"]):
            s_chain = []
            f_chain = []
            a_chain = []
            for position, draw in enumerate(indices):
                baseline = self.samples["baseline_hazard"][chain, draw]
                if path is None:
                    s, f, a = survival_curves(
                        eta[chain, position], baseline, self.data["baseline_breaks"], t, entry=entry
                    )
                else:
                    levels, s, f, a = survival_path_curves(
                        eta[chain, position], baseline, self.data["baseline_breaks"], t, **path
                    )
                s_chain.append(s)
                f_chain.append(f)
                a_chain.append(a)
            collected.append(s_chain)
            incidences.append(f_chain)
            hazards.append(a_chain)
        return np.asarray(collected), np.asarray(incidences), np.asarray(hazards), t, levels

    def predict_survival(
        self,
        Znew=None,
        Xnew=None,
        *,
        times,
        kind="survival",
        entry=0,
        cluster_new=None,
        random_slopes_new=None,
        probability=0.95,
        max_draws=None,
        seed=None,
    ):
        """Return survival probabilities or cause-specific hazards for fixed profiles.

        Curves describe subjects who survive to ``entry``. Survival probabilities
        include all causes. A hazard is a cause-specific event rate among subjects
        still at risk; it uses the cause selected by ``for_cause`` (the first cause
        by default). Beyond the last cutpoint, the final baseline hazard continues
        while each profile's predictors stay constant.
        """
        if kind not in ("survival", "hazard"):
            raise ValueError("kind must be survival or hazard")
        s, _, a, t, _ = self._survival_draws(
            Znew,
            Xnew,
            times=times,
            entry=entry,
            cluster_new=cluster_new,
            random_slopes_new=random_slopes_new,
            probability=probability,
            max_draws=max_draws,
            seed=seed,
        )
        return _survival_prediction(
            s if kind == "survival" else a[..., self._cause_index()], t, probability, kind
        )

    def predict_cumulative_incidence(
        self,
        Znew=None,
        Xnew=None,
        *,
        times,
        cause=None,
        entry=0,
        cluster_new=None,
        random_slopes_new=None,
        probability=0.95,
        max_draws=None,
        seed=None,
    ):
        """Return the probability of each cause occurring by each requested time.

        Describe subjects who survive to ``entry`` and account for all competing
        causes. ``cause=None`` returns all causes together; an integer selects one
        cause, numbered from one.
        """
        _, f, _, t, _ = self._survival_draws(
            Znew,
            Xnew,
            times=times,
            entry=entry,
            cluster_new=cluster_new,
            random_slopes_new=random_slopes_new,
            probability=probability,
            max_draws=max_draws,
            seed=seed,
        )
        if cause is not None:
            index = _integer(cause, "cause") - 1
            if index >= self.settings["n_causes"]:
                raise ValueError("cause exceeds fitted causes")
            return _survival_prediction(f[..., index], t, probability, "cumulative_incidence")
        return self._incidence_result(f, t, probability)

    def _incidence_result(self, values, times, probability):
        flat = values.reshape((-1,) + values.shape[2:])
        tail = (1 - probability) / 2
        low, high = np.quantile(flat, [tail, 1 - tail], axis=0)
        return CompetingRiskPrediction(
            *map(_freeze, (values, flat.mean(axis=0), flat.std(axis=0, ddof=1), low, high, times)),
            probability,
            tuple(range(1, self.settings["n_causes"] + 1)),
        )

    def predict_survival_path(
        self,
        Znew,
        Xnew=None,
        *,
        start,
        stop,
        subject_new,
        times,
        kind="survival",
        cause=None,
        cluster_new=None,
        random_slopes_new=None,
        probability=0.95,
        max_draws=None,
        seed=None,
    ):
        """Predict survival along supplied exposure and covariate histories.

        Predictors stay constant within each supplied interval. Requested times may
        not exceed the final stop; the last predictor row is never extended.
        Output row order follows first occurrence of subject_new. Repeated new
        cluster labels share one random-effect vector across intervals in each draw;
        omitted labels set random effects to zero.
        """
        if kind not in ("survival", "hazard", "cumulative_incidence"):
            raise ValueError("invalid path prediction kind")
        s, f, a, t, _ = self._survival_draws(
            Znew,
            Xnew,
            times=times,
            entry=0,
            cluster_new=cluster_new,
            random_slopes_new=random_slopes_new,
            probability=probability,
            max_draws=max_draws,
            seed=seed,
            path=dict(start=start, stop=stop, subject=subject_new),
        )
        if kind == "survival":
            return _survival_prediction(s, t, probability, kind)
        if kind == "cumulative_incidence" and cause is None:
            return self._incidence_result(f, t, probability)
        index = self._cause_index() if cause is None else _integer(cause, "cause") - 1
        if index >= self.settings["n_causes"]:
            raise ValueError("cause exceeds fitted causes")
        return _survival_prediction(
            f[..., index] if kind == "cumulative_incidence" else a[..., index], t, probability, kind
        )
