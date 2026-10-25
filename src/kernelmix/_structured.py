"""Noncentered inference for structured residuals, survival histories and GP bases."""

import math
import time
import warnings
from dataclasses import asdict

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import log_ndtr, ndtr

from ._basis import basis_matrix, prepare_basis
from ._clusters import canonical_label, cluster_mask
from ._generalized import _count, _log_proposal, _positive, elliptical_slice
from ._input import matrix, prepare_data
from ._likelihoods import negative_binomial_loglik, poisson_loglik
from ._structure import _standardized_residual, correlation_blocks, gaussian_loglik
from ._survival import extended_survival_loglik, prepare_extended_survival
from .priors import Prior, gamma_logpdf


def _original(value, n, name, *, two_dimensional=False, infinity=False):
    a = np.asarray(value, dtype=float)
    if two_dimensional and a.ndim == 1:
        a = a[:, None]
    shape_ok = a.ndim == 2 and len(a) == n if two_dimensional else a.shape == (n,)
    if not shape_ok or (not infinity and np.isinf(a).any()) or np.isneginf(a).any():
        raise ValueError(f"{name} must have one numeric row per original observation")
    return a.copy()


def _rank(a, name, *, no_constant=False):
    if not a.shape[1]:
        return
    check = np.column_stack((np.ones(len(a)), a)) if no_constant else a
    magnitude = np.max(np.abs(check), axis=0)
    if (magnitude == 0).any() or np.linalg.matrix_rank(check / magnitude) != check.shape[1]:
        suffix = " and must not contain an intercept in its column span" if no_constant else ""
        raise ValueError(f"{name} columns must be linearly independent{suffix}")


def _prepare(
    y,
    Z,
    X,
    *,
    family,
    standardize,
    intercept,
    missing,
    exposure_names,
    covariate_names,
    groups,
    cluster,
    offset,
    event,
    baseline_breaks,
    random_slopes,
    variance_design,
    correlation,
    observation_time,
    entry,
    upper,
    subject,
    n_causes,
):
    outcome = np.asarray(y, dtype=float)
    if outcome.ndim != 1:
        raise ValueError("y must be a one-dimensional vector")
    n = len(outcome)
    z = matrix(Z, "Z")
    x = np.empty((n, 0)) if X is None else matrix(X, "X")
    if len(z) != n or len(x) != n or np.isinf(outcome).any():
        raise ValueError("outcome, exposures and covariates must have matching finite rows")
    extra = {}
    for name, value, is_matrix, infinity in (
        ("random_slopes", random_slopes, True, False),
        ("variance_design", variance_design, True, False),
        ("offset", offset, False, False),
        ("event", event, False, False),
        ("entry", entry, False, False),
        ("upper", upper, False, True),
        ("observation_time", observation_time, False, False),
    ):
        if value is not None:
            extra[name] = _original(value, n, name, two_dimensional=is_matrix, infinity=infinity)
    if family != "survival" and any(
        v is not None for v in (entry, upper, subject, n_causes, event, baseline_breaks)
    ):
        raise ValueError('survival inputs require family="survival"')
    if family not in ("poisson", "negative_binomial") and offset is not None:
        raise ValueError("offset applies only to count models")
    if family != "gaussian" and (
        variance_design is not None or correlation is not None or observation_time is not None
    ):
        raise ValueError("variance_design and residual correlation apply only to Gaussian outcomes")
    if random_slopes is not None and cluster is None:
        raise ValueError("random_slopes requires cluster labels")
    if correlation is not None and cluster is None:
        raise ValueError("residual correlation requires cluster labels")
    if correlation is None and observation_time is not None:
        raise ValueError("observation_time requires a residual correlation model")
    keep = np.isfinite(outcome) & np.isfinite(z).all(axis=1) & np.isfinite(x).all(axis=1)
    keep &= cluster_mask(cluster, n)
    for a in extra.values():
        available = ~np.isnan(a)
        keep &= available.all(axis=1) if a.ndim == 2 else available
    dropped_subjects = []
    if subject is not None:
        labels = np.asarray(subject, dtype=object)
        keep &= cluster_mask(labels, n)
        by_subject = {}
        for i, label in enumerate(labels):
            if label is None or isinstance(label, (float, np.floating)) and np.isnan(label):
                continue
            by_subject.setdefault(canonical_label(label), []).append(i)
        for identity, rows in by_subject.items():
            if not keep[rows].all():
                keep[rows] = False
                dropped_subjects.append(identity[1])
    if missing not in ("raise", "drop"):
        raise ValueError("missing must be raise or drop")
    if missing == "raise" and not keep.all():
        raise ValueError(
            'missing input values; missing="drop" removes whole incomplete subject histories'
        )
    masked = outcome.copy()
    masked[~keep] = np.nan
    data = prepare_data(
        masked,
        Z,
        X,
        family="gaussian",
        standardize=standardize,
        intercept=intercept,
        missing=missing,
        exposure_names=exposure_names,
        covariate_names=covariate_names,
        groups=groups,
        cluster=cluster,
    )
    rows = data["rows"]
    data["y"] = outcome[rows].copy()
    extra = {key: a[rows].copy() for key, a in extra.items()}
    data["dropped_subject_levels"] = dropped_subjects
    data["variance_design"] = extra.get("variance_design", np.empty((len(rows), 0)))
    _rank(data["variance_design"], "variance_design", no_constant=True)
    slopes = extra.get("random_slopes", np.empty((len(rows), 0)))
    data["random_design"] = (
        np.column_stack((np.ones(len(rows)), slopes))
        if cluster is not None
        else np.empty((len(rows), 0))
    )
    _rank(data["random_design"], "random coefficient design")
    if family == "binomial" and not np.isin(data["y"], [0, 1]).all():
        raise ValueError("binomial outcomes must be zero or one")
    if family in ("poisson", "negative_binomial"):
        if (data["y"] < 0).any() or (data["y"] != np.floor(data["y"])).any():
            raise ValueError("count outcomes must be nonnegative integers")
        data["offset"] = extra.get("offset", np.zeros(len(rows)))
    if family == "survival":
        if event is None:
            raise ValueError("survival outcomes require event indicators or cause codes")
        labels = None if subject is None else np.asarray(subject, dtype=object)[rows]
        data.update(
            prepare_extended_survival(
                data["y"],
                extra["event"],
                entry=extra.get("entry"),
                upper=extra.get("upper"),
                subject=labels,
                n_causes=n_causes,
                baseline_breaks=baseline_breaks,
            )
        )
    data["observation_time"] = extra.get("observation_time")
    if correlation is None:
        data["correlation_matrix"], kind = None, "independent"
    elif isinstance(correlation, str):
        if correlation not in ("ar1", "exponential"):
            raise ValueError("correlation must be ar1, exponential, a matrix, or None")
        data["correlation_matrix"], kind = None, correlation
        if observation_time is None:
            raise ValueError("parametric residual correlation requires observation_time")
    else:
        corr = np.asarray(correlation, dtype=float)
        if corr.shape != (n, n) or not np.isfinite(corr).all():
            raise ValueError("fixed correlation must be a finite original-row square matrix")
        data["correlation_matrix"], kind = corr[np.ix_(rows, rows)].copy(), "fixed"
    return data, kind


def _random_factor(scale_raw, correlation_raw, bound):
    """Proper induced correlation prior and bounded uniform standard deviations."""
    k, q = scale_raw.shape
    result = np.zeros((k, q, q))
    for cause in range(k):
        lower = np.eye(q)
        lower[np.tril_indices(q, -1)] = correlation_raw[cause]
        lower /= np.linalg.norm(lower, axis=1)[:, None]
        sd = bound * ndtr(scale_raw[cause])
        if not np.isfinite(sd).all() or (sd <= 0).any():
            raise FloatingPointError("random-effect scales reached a floating-point boundary")
        result[cause] = sd[:, None] * lower
    return result


class _Layout:
    def __init__(self):
        self.parts = {}
        self.size = 0

    def add(self, name, shape):
        size = math.prod(shape)
        self.parts[name] = (slice(self.size, self.size + size), tuple(shape))
        self.size += size

    def get(self, white, name):
        sl, shape = self.parts[name]
        return white[sl].reshape(shape)


def fit_structured(
    y,
    Z,
    X=None,
    *,
    family="gaussian",
    draws=1000,
    warmup=1000,
    chains=4,
    thin=1,
    seed=None,
    standardize=True,
    intercept=True,
    selection="none",
    groups=None,
    prior=None,
    r_fixed=None,
    missing="raise",
    exposure_names=None,
    covariate_names=None,
    initial=None,
    proposal_scales=None,
    cluster=None,
    offset=None,
    event=None,
    baseline_breaks=None,
    kernel="gaussian",
    kernel_alpha=1,
    approximation="dense",
    n_components=32,
    knots=None,
    random_slopes=None,
    random_covariance=None,
    random_scale=2,
    correlation=None,
    observation_time=None,
    correlation_parameter=None,
    variance_design=None,
    entry=None,
    upper=None,
    subject=None,
    n_causes=None,
):
    """Fit the documented structured model with a dense or finite-basis GP."""
    started = time.perf_counter()
    if family not in ("gaussian", "binomial", "poisson", "negative_binomial", "survival"):
        raise ValueError("unknown outcome family")
    draws, warmup = _count(draws, "draws", 2), _count(warmup, "warmup", 0)
    chains, thin = _count(chains, "chains", 1), _count(thin, "thin", 1)
    if seed is not None:
        seed = _count(seed, "seed", 0)
    if selection not in ("none", "component", "group") or (
        (selection == "group") != (groups is not None)
    ):
        raise ValueError(
            "selection must be none/component/group; groups are required only for group selection"
        )
    defaults = dict(
        lambda_mean=1.0,
        lambda_sd=0.5,
        cluster_lambda_mean=1.0,
        cluster_lambda_sd=0.5,
        sigma2_shape=2.0,
        sigma2_rate=1.0,
    )
    if prior is None:
        prior = Prior(**defaults)
    elif isinstance(prior, dict):
        prior = Prior(**{**defaults, **prior})
    if not isinstance(prior, Prior):
        raise ValueError("prior must be Prior, a dictionary or None")
    data, corr_kind = _prepare(
        y,
        Z,
        X,
        family=family,
        standardize=standardize,
        intercept=intercept,
        missing=missing,
        exposure_names=exposure_names,
        covariate_names=covariate_names,
        groups=groups,
        cluster=cluster,
        offset=offset,
        event=event,
        baseline_breaks=baseline_breaks,
        random_slopes=random_slopes,
        variance_design=variance_design,
        correlation=correlation,
        observation_time=observation_time,
        entry=entry,
        upper=upper,
        subject=subject,
        n_causes=n_causes,
    )
    n, m = data["Z"].shape
    p = data["X"].shape[1]
    k = data.get("n_causes", 1)
    g = len(data["cluster_levels"])
    q = data["random_design"].shape[1]
    random_scale = _positive(random_scale, "random_scale")
    if random_covariance is not None and not g:
        raise ValueError("random_covariance requires cluster labels")
    fixed_random = None
    if random_covariance is not None:
        fixed_random = np.asarray(random_covariance, dtype=float)
        if (
            fixed_random.shape != (q, q)
            or not np.isfinite(fixed_random).all()
            or not np.allclose(fixed_random, fixed_random.T, rtol=0, atol=1e-12)
        ):
            raise ValueError(
                "random_covariance must be a finite symmetric matrix matching the random design"
            )
        try:
            np.linalg.cholesky(fixed_random)
        except np.linalg.LinAlgError as exc:
            raise ValueError("random_covariance must be positive definite") from exc
    data["fixed_random_covariance"] = None if fixed_random is None else fixed_random.copy()
    corr_inferred = corr_kind in ("ar1", "exponential") and correlation_parameter is None
    if corr_kind in ("independent", "fixed") and correlation_parameter is not None:
        raise ValueError("correlation_parameter applies only to ar1/exponential correlation")
    initial_corr = 0.0 if corr_kind == "ar1" else 1.0 if corr_kind == "exponential" else None
    if correlation_parameter is not None:
        if (
            isinstance(correlation_parameter, (bool, np.bool_))
            or not np.isscalar(correlation_parameter)
            or not np.isfinite(correlation_parameter)
        ):
            raise ValueError("correlation_parameter must be a finite scalar")
        initial_corr = float(correlation_parameter)
    if corr_kind != "independent":
        correlation_blocks(
            data["cluster_codes"],
            data["observation_time"],
            corr_kind,
            initial_corr,
            data["correlation_matrix"],
        )
    sequence = np.random.SeedSequence(seed)
    streams = sequence.spawn(chains + 1)
    transformed_knots = None
    if knots is not None:
        raw = np.asarray(knots, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != m or not np.isfinite(raw).all():
            raise ValueError("knots must have finite rows in original exposure units")
        transformed_knots = (raw - data["z_center"]) / data["z_scale"]
    basis = prepare_basis(
        data["Z"],
        approximation=approximation,
        kernel=kernel,
        alpha=kernel_alpha,
        n_components=n_components,
        knots=transformed_knots,
        rng=np.random.default_rng(streams[0]),
    )
    b = basis["n_features"]
    mode = basis["approximation"]
    data["basis_meta"] = {
        name: value for name, value in basis.items() if not isinstance(value, np.ndarray)
    }
    for name, value in basis.items():
        if isinstance(value, np.ndarray):
            data["basis_" + name] = value.copy()
    anchor_rows = n if mode == "dense" else len(basis["knots"]) if mode == "pp" else 0
    layout = _Layout()
    layout.add("gp", (k, b))
    layout.add("beta", (k, p))
    layout.add("random", (k, g, q))
    layout.add("gamma", (data["variance_design"].shape[1],))
    if g and fixed_random is None:
        layout.add("random_scale", (k, q))
        layout.add("random_correlation", (k, q * (q - 1) // 2))
    if corr_inferred:
        layout.add("correlation", (1,))
    shapes = {
        "beta": (k, p),
        "r": (m,),
        "lambda": (k,),
        "sigma2": (),
        "delta": (m,),
        "pi": (),
        "gp_weight": (k, b),
    }
    if anchor_rows:
        shapes["gp_anchor"] = (k, anchor_rows)
    if g:
        shapes.update(random_coeff=(k, g, q), random_covariance=(k, q, q))
    if data["variance_design"].shape[1]:
        shapes["gamma"] = (data["variance_design"].shape[1],)
    if corr_kind in ("ar1", "exponential"):
        shapes["correlation_parameter"] = ()
    if family == "negative_binomial":
        shapes["dispersion"] = ()
    if family == "survival":
        shapes["baseline_hazard"] = (k, len(data["baseline_breaks"]) + 1)
    saved_elements = chains * draws * sum(math.prod(shape) for shape in shapes.values())
    if 8 * (saved_elements + 8 * n * b + 4 * layout.size) > 512 * 1024**2:
        raise ValueError(
            "structured fit exceeds the 512 MiB working-memory limit; reduce saved states or basis size"
        )
    samples = {
        name: np.empty((chains, draws) + shape, dtype=bool if name == "delta" else float)
        for name, shape in shapes.items()
    }
    fixed = None if r_fixed is None else np.asarray(r_fixed, dtype=float)
    if fixed is not None and (
        fixed.shape != (m,)
        or not np.isfinite(fixed).all()
        or (fixed < 0).any()
        or selection != "none"
    ):
        raise ValueError(
            "r_fixed must be nonnegative with one entry per exposure and disallows selection"
        )
    units = (
        [np.array([j]) for j in range(m)]
        if selection == "component"
        else [
            np.flatnonzero(np.asarray(data["groups"]) == label)
            for label in dict.fromkeys(data["groups"])
        ]
        if selection == "group"
        else []
    )
    scales = {"r": 0.3, "lambda": 0.3}
    if family == "negative_binomial":
        scales["dispersion"] = 0.3
    if family == "survival":
        scales["baseline_hazard"] = 0.3
    if proposal_scales is not None:
        if not isinstance(proposal_scales, dict) or set(proposal_scales) - set(scales):
            raise ValueError(f"proposal_scales accepts only {list(scales)} for this model")
        scales.update(proposal_scales)
    scales = {name: _positive(value, f"proposal_scales[{name}]") for name, value in scales.items()}
    if initial is None:
        initial = {}
    allowed = {"beta", "gp_weight", "r", "delta", "pi", "lambda", "sigma2"}
    if family == "negative_binomial":
        allowed.add("dispersion")
    if family == "survival":
        allowed.add("baseline_hazard")
    if not isinstance(initial, dict) or set(initial) - allowed:
        raise ValueError(f"initial accepts only {sorted(allowed)} for this model")
    messages = []
    if prior.lambda_mean / prior.lambda_sd**2 <= 2:
        messages.append(
            "The supplied GP amplitude prior does not guarantee finite second exp-Gaussian moments; finite-chain means do not prove moment existence."
        )
    if family == "gaussian" and prior.sigma2_shape <= 1:
        messages.append("The supplied inverse-gamma residual prior has no finite mean.")
    if family == "survival":
        if intercept:
            messages.append(
                "Survival intercept and baseline scale are separated by their priors; consider intercept=False."
            )
        if not (data["event"] != 0).any():
            messages.append("No observed events; survival estimates depend strongly on the priors.")
        if (data["entry"] > 0).any():
            messages.append(
                "Latent priors describe the included entrant cohort, not an inception-population frailty distribution."
            )
    if g and (g < q + 1 or all(np.sum(data["cluster_codes"] == j) == 1 for j in range(g))):
        messages.append(
            "Few repeated clusters for random covariance estimation; variance components may be weakly identified."
        )
    if mode != "dense":
        messages.append(
            "Inference conditions on the selected finite GP basis; it is not exact dense-GP inference."
        )
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    chain_info = []
    for chain, stream in enumerate(streams[1:]):
        rng = np.random.default_rng(stream)
        delta = np.ones(m, dtype=bool)
        if selection != "none":
            delta[:] = False
            for members in units:
                if rng.random() < 0.5:
                    delta[rng.choice(members)] = True
        if "delta" in initial:
            supplied = np.asarray(initial["delta"])
            if supplied.shape != (m,) or not np.isin(supplied, [0, 1]).all():
                raise ValueError("initial delta must be zero/one per exposure")
            delta = supplied.astype(bool)
        if fixed is not None:
            if "delta" in initial and not np.array_equal(delta, fixed > 0):
                raise ValueError("initial delta conflicts with r_fixed")
            delta = fixed > 0
        if selection == "none" and fixed is None and not delta.all():
            raise ValueError("all exposures must be active without selection")
        if selection == "group" and any(delta[u].sum() > 1 for u in units):
            raise ValueError("at most one exposure per group may be active")
        r = (
            fixed.copy()
            if fixed is not None
            else np.array([prior.sample_r(rng) if active else 0.0 for active in delta])
        )
        if "r" in initial:
            supplied = np.asarray(initial["r"], dtype=float)
            if fixed is not None and not np.array_equal(supplied, fixed):
                raise ValueError("initial r conflicts with r_fixed")
            r = supplied.copy()
        if (
            r.shape != (m,)
            or not np.isfinite(r).all()
            or (r[~delta] != 0).any()
            or (fixed is None and any(not np.isfinite(prior.log_r(v)) for v in r[delta]))
        ):
            raise ValueError("initial relevance is inconsistent with selection or its prior")
        white = np.zeros(layout.size)
        for name, target, scale in [("beta", "beta", prior.beta_sd), ("gp_weight", "gp", 1.0)]:
            if name in initial:
                a = np.asarray(initial[name], dtype=float)
                sl, shape = layout.parts[target]
                if k == 1 and a.shape == shape[1:]:
                    a = a[None, :]
                if a.shape != shape or not np.isfinite(a).all():
                    raise ValueError(f"initial {name} has an invalid shape")
                white[sl] = a.ravel() / scale
        amplitude = np.asarray(initial.get("lambda", np.full(k, prior.lambda_mean)), dtype=float)
        if amplitude.ndim == 0:
            amplitude = np.full(k, float(amplitude))
        if amplitude.shape != (k,) or not np.isfinite(amplitude).all() or (amplitude <= 0).any():
            raise ValueError("initial lambda must be positive per cause")
        sigma2 = _positive(
            initial.get(
                "sigma2", max(float(np.var(data["y"])), 0.1) if family == "gaussian" else 1.0
            ),
            "initial sigma2",
        )
        if family != "gaussian" and sigma2 != 1:
            raise ValueError("non-Gaussian sigma2 is fixed at one")
        phi = (
            _positive(
                initial.get("dispersion", prior.dispersion_shape / prior.dispersion_rate),
                "initial dispersion",
            )
            if family == "negative_binomial"
            else None
        )
        baseline = np.asarray(
            initial.get(
                "baseline_hazard",
                np.full(
                    shapes.get("baseline_hazard", (0, 0)),
                    prior.baseline_shape / prior.baseline_rate,
                ),
            ),
            dtype=float,
        )
        if family == "survival" and (
            baseline.shape != shapes["baseline_hazard"]
            or not np.isfinite(baseline).all()
            or (baseline <= 0).any()
        ):
            raise ValueError(
                "initial baseline_hazard must be positive with shape (causes, intervals)"
            )
        pi = float(initial.get("pi", 1.0 if selection == "none" else 0.5))
        if (
            not np.isfinite(pi)
            or (selection == "none" and pi != 1)
            or (selection != "none" and not 0 < pi < 1)
        ):
            raise ValueError("initial pi is inconsistent with selection")
        factor = basis_matrix(data["Z"], r, basis)

        def components(w, f=None, amp=None):
            f = factor if f is None else f
            amp = amplitude if amp is None else amp
            gp = (f @ layout.get(w, "gp").T) * np.sqrt(amp)[None, :]
            beta = layout.get(w, "beta") * prior.beta_sd
            eta = gp + data["X"] @ beta.T
            cov = np.empty((k, 0, 0))
            coeff = np.empty((k, 0, 0))
            if g:
                lower = (
                    np.repeat(np.linalg.cholesky(fixed_random)[None], k, axis=0)
                    if fixed_random is not None
                    else _random_factor(
                        layout.get(w, "random_scale"),
                        layout.get(w, "random_correlation"),
                        random_scale,
                    )
                )
                coeff = np.einsum("kgq,krq->kgr", layout.get(w, "random"), lower)
                cov = lower @ lower.transpose(0, 2, 1)
                eta += np.einsum(
                    "nkq,nq->nk",
                    coeff[:, data["cluster_codes"], :].transpose(1, 0, 2),
                    data["random_design"],
                )
            gamma = layout.get(w, "gamma")
            logvar = data["variance_design"] @ gamma
            corr = (
                (
                    float(np.tanh(layout.get(w, "correlation")[0]))
                    if corr_kind == "ar1"
                    else math.exp(float(layout.get(w, "correlation")[0]))
                )
                if corr_inferred
                else initial_corr
            )
            blocks = (
                correlation_blocks(
                    data["cluster_codes"],
                    data["observation_time"],
                    corr_kind,
                    corr,
                    data["correlation_matrix"],
                )
                if corr_kind != "independent"
                else []
            )
            return gp, beta, eta, coeff, cov, gamma, logvar, corr, blocks

        def likelihood(w, f=None, amp=None):
            try:
                _, _, eta, _, _, _, logvar, _, blocks = components(w, f, amp)
                if not np.isfinite(eta).all() or not np.isfinite(logvar).all():
                    return -np.inf
                if family == "gaussian":
                    return gaussian_loglik(data["y"], eta[:, 0], sigma2, logvar, blocks)
                if family == "binomial":
                    return float(log_ndtr((2 * data["y"] - 1) * eta[:, 0]).sum())
                if family == "poisson":
                    return poisson_loglik(data["y"], eta[:, 0] + data["offset"])
                if family == "negative_binomial":
                    return negative_binomial_loglik(data["y"], eta[:, 0] + data["offset"], phi)
                return extended_survival_loglik(eta, baseline, data)
            except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
                return -np.inf

        current = likelihood(white)
        if not np.isfinite(current):
            raise ValueError("initial structured state has a nonfinite likelihood")
        counters = {
            name: dict(proposed=0, accepted=0, warmup_proposed=0, warmup_accepted=0)
            for name in [*scales, "selection"]
        }
        evaluations = 1
        saved = 0

        def accept(name, old_ll, new_ll, extra, warm):
            record = counters[name]
            prefix = "warmup_" if warm else ""
            record[prefix + "proposed"] += 1
            ok = np.isfinite(new_ll) and math.log1p(-float(rng.random())) < new_ll - old_ll + extra
            if ok:
                record[prefix + "accepted"] += 1
            return ok

        for iteration in range(warmup + draws * thin):
            warm = iteration < warmup
            white, current, calls = elliptical_slice(white, likelihood, rng, current_loglik=current)
            evaluations += calls
            for cause in range(k):
                candidate, jacobian = _log_proposal(amplitude[cause], scales["lambda"], rng)
                if np.isfinite(candidate) and candidate > 0:
                    proposed = amplitude.copy()
                    proposed[cause] = candidate
                    new_ll = likelihood(white, amp=proposed)
                    evaluations += 1
                    extra = (
                        gamma_logpdf(candidate, prior.lambda_mean, prior.lambda_sd)
                        - gamma_logpdf(amplitude[cause], prior.lambda_mean, prior.lambda_sd)
                        + jacobian
                    )
                    if accept("lambda", current, new_ll, extra, warm):
                        amplitude, current = proposed, new_ll
            if fixed is None:
                for j in np.flatnonzero(delta):
                    candidate, jacobian = _log_proposal(r[j], scales["r"], rng)
                    if not np.isfinite(prior.log_r(candidate)):
                        continue
                    proposed = r.copy()
                    proposed[j] = candidate
                    try:
                        new_factor = basis_matrix(data["Z"], proposed, basis)
                    except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
                        continue
                    new_ll = likelihood(white, f=new_factor)
                    evaluations += 1
                    if accept(
                        "r",
                        current,
                        new_ll,
                        prior.log_r(candidate) - prior.log_r(r[j]) + jacobian,
                        warm,
                    ):
                        r, factor, current = proposed, new_factor, new_ll
            for members in units:
                proposed_delta = delta.copy()
                proposed_delta[members] = False
                proposed_r = r.copy()
                proposed_r[members] = 0
                if rng.random() < pi:
                    j = int(rng.choice(members))
                    proposed_delta[j] = True
                    proposed_r[j] = prior.sample_r(rng)
                if np.array_equal(proposed_r, r):
                    continue
                new_factor = basis_matrix(data["Z"], proposed_r, basis)
                new_ll = likelihood(white, f=new_factor)
                evaluations += 1
                # The proposal draws this unit from its conditional selection/relevance prior.
                if accept("selection", current, new_ll, 0.0, warm):
                    r, delta, factor, current = proposed_r, proposed_delta, new_factor, new_ll
            if units:
                active = sum(bool(delta[u].any()) for u in units)
                pi = float(
                    rng.beta(
                        prior.inclusion_alpha + active, prior.inclusion_beta + len(units) - active
                    )
                )
            if family == "gaussian":
                _, _, eta, _, _, _, logvar, _, blocks = components(white)
                residual = _standardized_residual(data["y"], eta[:, 0], 1.0, logvar)
                if blocks:
                    quadratic = 0.0
                    for indices, _, lower in blocks:
                        transformed = solve_triangular(
                            lower, residual[indices], lower=True, check_finite=False
                        )
                        quadratic += float(transformed @ transformed)
                else:
                    quadratic = float(residual @ residual)
                sigma2 = 1 / float(
                    rng.gamma(prior.sigma2_shape + n / 2, 1 / (prior.sigma2_rate + quadratic / 2))
                )
                current = likelihood(white)
                evaluations += 1
            if family == "negative_binomial":
                candidate, jacobian = _log_proposal(phi, scales["dispersion"], rng)
                old = phi
                phi = candidate
                new_ll = likelihood(white)
                evaluations += 1
                extra = (
                    (
                        (prior.dispersion_shape - 1) * (math.log(candidate) - math.log(old))
                        - prior.dispersion_rate * (candidate - old)
                        + jacobian
                    )
                    if np.isfinite(candidate) and candidate > 0
                    else -np.inf
                )
                if accept("dispersion", current, new_ll, extra, warm):
                    current = new_ll
                else:
                    phi = old
            if family == "survival":
                for cause in range(k):
                    for j in range(baseline.shape[1]):
                        old = baseline[cause, j]
                        candidate, jacobian = _log_proposal(old, scales["baseline_hazard"], rng)
                        baseline[cause, j] = candidate
                        new_ll = likelihood(white)
                        evaluations += 1
                        extra = (
                            (
                                (prior.baseline_shape - 1) * (math.log(candidate) - math.log(old))
                                - prior.baseline_rate * (candidate - old)
                                + jacobian
                            )
                            if np.isfinite(candidate) and candidate > 0
                            else -np.inf
                        )
                        if accept("baseline_hazard", current, new_ll, extra, warm):
                            current = new_ll
                        else:
                            baseline[cause, j] = old
            if not np.isfinite(current):
                raise FloatingPointError("structured state left finite likelihood support")
            if not warm and (iteration - warmup + 1) % thin == 0:
                gp, beta, _, coeff, covariance, gamma, _, corr, _ = components(white)
                values = dict(
                    beta=beta,
                    r=r,
                    delta=delta,
                    pi=pi,
                    sigma2=sigma2,
                    gp_weight=layout.get(white, "gp"),
                )
                values["lambda"] = amplitude
                if anchor_rows:
                    anchor = (
                        gp.T
                        if mode == "dense"
                        else (
                            basis_matrix(basis["knots"], r, basis)
                            @ layout.get(white, "gp").T
                            * np.sqrt(amplitude)[None, :]
                        ).T
                    )
                    values["gp_anchor"] = anchor
                if g:
                    values.update(random_coeff=coeff, random_covariance=covariance)
                if gamma.size:
                    values["gamma"] = gamma
                if corr_kind in ("ar1", "exponential"):
                    values["correlation_parameter"] = corr
                if family == "negative_binomial":
                    values["dispersion"] = phi
                if family == "survival":
                    values["baseline_hazard"] = baseline
                for name, value in values.items():
                    samples[name][chain, saved] = value
                saved += 1
        for counter in counters.values():
            for prefix in ("", "warmup_"):
                count = counter[prefix + "proposed"]
                counter[prefix + "rate"] = counter[prefix + "accepted"] / count if count else None
        chain_info.append(
            dict(
                chain=chain,
                acceptance=counters,
                likelihood_evaluations=evaluations,
                seed_spawn_key=list(stream.spawn_key),
            )
        )
    settings = dict(
        engine="structured",
        family=family,
        draws=draws,
        warmup=warmup,
        chains=chains,
        thin=thin,
        seed=seed,
        standardize=bool(standardize),
        intercept=bool(intercept),
        selection=selection,
        prior=asdict(prior),
        r_fixed=None if fixed is None else fixed.tolist(),
        clustered=bool(g),
        random_scale=random_scale,
        random_covariance_fixed=fixed_random is not None,
        correlation=corr_kind,
        correlation_fixed=not corr_inferred,
        correlation_parameter=correlation_parameter,
        kernel=kernel,
        kernel_alpha=float(kernel_alpha),
        approximation=mode,
        n_causes=k,
        active_cause=1,
        amplitude_convention="direct_tau",
        offset_used=offset is not None,
        proposal_scales=scales,
        beta_prior="normal",
        entrant_cohort_conditional=True,
    )
    info = dict(
        runtime_seconds=time.perf_counter() - started,
        chains=chain_info,
        algorithm="noncentered ESS/log-MH/Gaussian-variance-Gibbs",
        proposal_adaptation=False,
        seed_entropy=int(sequence.entropy),
        warnings=messages,
    )
    from .structured_result import StructuredFit

    return StructuredFit(data, samples, settings, info)
