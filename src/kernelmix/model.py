"""Bayesian kernel machine regression (BKMR) fitting and sampler selection.

The default Gaussian/probit target follows bkmr 0.2.2, with different
Metropolis-Hastings proposals. Other supported inputs select the generalized
or structured sampler.
No global random state is used and chains run sequentially.
"""

import math
import time
from dataclasses import asdict

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.stats import truncnorm

from ._input import prepare_data
from ._kernel import covariance_components
from .priors import Prior, gamma_logpdf


def _count(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _initial(initial, data, prior, family, selection, fixed, rng):
    n, m = data["Z"].shape
    p = data["X"].shape[1]
    if initial is None:
        initial = {}
    if not isinstance(initial, dict):
        raise ValueError("initial must be a dictionary shared by chains, or None")
    allowed = {"beta", "r", "lambda", "sigma2", "delta", "pi", "ystar"}
    if set(initial) - allowed:
        raise ValueError(f"unknown initial parameters: {sorted(set(initial) - allowed)}")
    delta = np.ones(m, dtype=bool)
    if selection == "group":
        delta[:] = False
        for label in dict.fromkeys(data["groups"]):
            members = np.flatnonzero(np.asarray(data["groups"]) == label)
            delta[rng.choice(members)] = True
    if "delta" in initial:
        d = np.asarray(initial["delta"])
        if d.shape != (m,) or not np.isin(d, [0, 1]).all():
            raise ValueError("initial delta must be boolean/0-1 with one entry per exposure")
        delta = d.astype(bool)
    if selection == "none" and not delta.all():
        raise ValueError("initial delta must be all true without selection")
    if selection == "group":
        for label in dict.fromkeys(data["groups"]):
            if delta[np.asarray(data["groups"]) == label].sum() > 1:
                raise ValueError("initial delta includes multiple exposures in one group")
    if fixed is not None:
        if "r" in initial and not np.array_equal(np.asarray(initial["r"]), fixed):
            raise ValueError("initial r conflicts with r_fixed")
        r = fixed.copy()
        delta = r > 0
    elif "r" in initial:
        r = np.asarray(initial["r"], dtype=float).copy()
        if r.shape != (m,) or not np.isfinite(r).all() or (r < 0).any():
            raise ValueError("initial r must be a finite nonnegative exposure vector")
        if (r[~delta] != 0).any() or any(not np.isfinite(prior.log_r(x)) for x in r[delta]):
            raise ValueError(
                "initial r must be zero when excluded, and inside the positive prior support when active"
            )
    else:
        r = np.array([prior.sample_r(rng) if active else 0.0 for active in delta])
    beta = np.asarray(initial.get("beta", np.zeros(p)), dtype=float).copy()
    if beta.shape != (p,) or not np.isfinite(beta).all():
        raise ValueError("initial beta must be finite with one entry per design column")
    lambda_ = _positive(initial.get("lambda", prior.lambda_mean), "initial lambda")
    default_variance = max(float(np.var(data["y"])), 0.1)
    sigma2 = _positive(
        initial.get("sigma2", 1.0 if family == "binomial" else default_variance), "initial sigma2"
    )
    if family == "binomial" and sigma2 != 1:
        raise ValueError("binomial sigma2 is fixed at 1")
    pi = _positive(initial.get("pi", 1.0 if selection == "none" else 0.5), "initial pi")
    if selection == "none" and pi != 1 or selection != "none" and not pi < 1:
        raise ValueError(
            "initial pi must equal 1 without selection, or lie in (0,1) with selection"
        )
    latent = None
    if family == "binomial":
        latent = np.asarray(
            initial.get("ystar", np.where(data["y"] == 1, 0.5, -0.5)), dtype=float
        ).copy()
        if (
            latent.shape != (n,)
            or not np.isfinite(latent).all()
            or not np.all(np.where(data["y"] == 1, latent > 0, latent < 0))
        ):
            raise ValueError("initial ystar must be finite and have the observed binary signs")
    elif "ystar" in initial:
        raise ValueError("initial ystar is only supported for binomial outcomes")
    return beta, r, lambda_, sigma2, delta, pi, latent


def _log_likelihood(lower, residual, sigma2):
    whitened = solve_triangular(lower, residual, lower=True, check_finite=False)
    return -np.log(np.diag(lower)).sum() - float(whitened @ whitened) / (2 * sigma2)


def _latent_sweep(y, latent, mean, lower, rng):
    # Coordinate Gibbs for the marginal truncated multivariate normal. Independent
    # univariate draws with marginal sd would discard the GP correlations.
    precision = cho_solve((lower, True), np.eye(len(y)), check_finite=False)
    residual = latent - mean
    for i in rng.permutation(len(y)):
        conditional_mean = (
            mean[i] - (precision[i] @ residual - precision[i, i] * residual[i]) / precision[i, i]
        )
        sd = math.sqrt(1 / precision[i, i])
        boundary = -conditional_mean / sd
        a, b = (boundary, np.inf) if y[i] == 1 else (-np.inf, boundary)
        latent[i] = truncnorm.rvs(a, b, loc=conditional_mean, scale=sd, random_state=rng)
        residual[i] = latent[i] - mean[i]
    return latent


def _check_probit_design(y, design):
    """Reject separated flat-prior designs using a column-unit invariant LP.

    Positive column rescaling preserves the set of attainable linear
    predictors. Only this diagnostic uses rescaled columns; the model and its
    coefficient parameterization retain the caller's original design.
    """
    if not design.shape[1]:
        return
    from scipy.optimize import linprog

    scales = np.max(np.abs(design), axis=0)
    if not np.isfinite(scales).all() or (scales == 0).any():
        raise ValueError("probit adjustment design must have finite nonzero columns")
    signed_design = (2 * y - 1)[:, None] * (design / scales)
    # In the signed-margin cone, a nonzero predictor with nonnegative margins
    # has positive summed margin. The coefficient box bounds the optimization
    # without excluding such a direction. Scaling first prevents solver
    # tolerances and the objective cutoff from depending on covariate units.
    separation = linprog(
        -signed_design.sum(axis=0),
        A_ub=-signed_design,
        b_ub=np.zeros(len(y)),
        bounds=[(-1, 1)] * design.shape[1],
        method="highs",
    )
    if not separation.success:
        raise ValueError("could not establish nonseparation for the flat probit beta prior")
    if -separation.fun > 1e-8:
        raise ValueError(
            "separated binomial adjustment design has an improper posterior under the flat beta prior"
        )


def fit(
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
    """Fit Bayesian kernel machine regression with chains run one after another.

    Rows of Z are exposure profiles z; rows of X are adjustment covariates x.
    The exposure function h can describe nonlinear associations and interactions;
    beta gives the linear adjustment coefficients. A kernel describes similarity
    between exposure profiles. A Gaussian process (GP) prior uses it to model h.
    Relevance r_m controls exposure m's contribution
    to that similarity. Group selection permits at most one active exposure in
    each included group.

    ``draws`` counts saved states per chain after ``warmup`` and ``thin``.
    The probit model fixes residual variance at one. Dense calculations store
    matrices with one row and column per retained observation and require roughly
    cubic work in that observation count. A successful fit does not prove that
    the chains have converged.

    For the default Gaussian/probit sampler, ``initial`` accepts beta, r,
    lambda, sigma2, delta and pi; probit also accepts ystar. ``proposal_scales``
    sets fixed log-scale proposal standard deviations for r and lambda, each
    0.3 by default. Other samplers' keys and defaults are in the model guides.
    ``cluster`` adds a random intercept shared by observations with the same
    label. ``offset`` is a count log offset; survival uses event and baseline_breaks.

    The paper's tau is the variance multiplying the kernel. The default
    Gaussian/probit sampler stores ``lambda=tau/sigma2`` and ``lambda_cluster``
    as the random-intercept variance divided by sigma2. Default count/survival
    samplers store those variances directly and use normal coefficient priors.

    Alternative kernels, finite GP bases, random slopes, residual correlation,
    variance regression, extended survival inputs or nondefault structured
    settings (such as random_scale) select the structured sampler. It stores
    ``lambda=tau`` directly and uses separate priors. The structured-model guide
    defines these inputs, initialization shapes and prediction requirements.
    """
    if not isinstance(kernel, str) or kernel not in ("gaussian", "matern32", "rational_quadratic"):
        raise ValueError("kernel must be gaussian, matern32 or rational_quadratic")
    _positive(kernel_alpha, "kernel_alpha")
    if kernel != "rational_quadratic" and kernel_alpha != 1:
        raise ValueError("kernel_alpha applies only to the rational-quadratic kernel")
    if not isinstance(approximation, str) or approximation not in ("dense", "pp", "rff"):
        raise ValueError("approximation must be dense, pp or rff")
    _count(n_components, "n_components", 1)
    _positive(random_scale, "random_scale")
    if random_scale != 2 and cluster is None:
        raise ValueError("random_scale requires cluster labels")
    extended = (
        kernel != "gaussian"
        or kernel_alpha != 1
        or approximation != "dense"
        or n_components != 32
        or random_scale != 2
        or any(
            value is not None
            for value in (
                knots,
                random_slopes,
                random_covariance,
                correlation,
                observation_time,
                correlation_parameter,
                variance_design,
                entry,
                upper,
                subject,
                n_causes,
            )
        )
    )
    if extended:
        from ._structured import fit_structured

        return fit_structured(
            y,
            Z,
            X,
            family=family,
            draws=draws,
            warmup=warmup,
            chains=chains,
            thin=thin,
            seed=seed,
            standardize=standardize,
            intercept=intercept,
            selection=selection,
            groups=groups,
            prior=prior,
            r_fixed=r_fixed,
            missing=missing,
            exposure_names=exposure_names,
            covariate_names=covariate_names,
            initial=initial,
            proposal_scales=proposal_scales,
            cluster=cluster,
            offset=offset,
            event=event,
            baseline_breaks=baseline_breaks,
            kernel=kernel,
            kernel_alpha=kernel_alpha,
            approximation=approximation,
            n_components=n_components,
            knots=knots,
            random_slopes=random_slopes,
            random_covariance=random_covariance,
            random_scale=random_scale,
            correlation=correlation,
            observation_time=observation_time,
            correlation_parameter=correlation_parameter,
            variance_design=variance_design,
            entry=entry,
            upper=upper,
            subject=subject,
            n_causes=n_causes,
        )
    if family in ("poisson", "negative_binomial", "survival"):
        from ._generalized import fit_generalized

        return fit_generalized(
            y,
            Z,
            X,
            family=family,
            draws=draws,
            warmup=warmup,
            chains=chains,
            thin=thin,
            seed=seed,
            standardize=standardize,
            intercept=intercept,
            selection=selection,
            groups=groups,
            prior=prior,
            r_fixed=r_fixed,
            missing=missing,
            exposure_names=exposure_names,
            covariate_names=covariate_names,
            initial=initial,
            proposal_scales=proposal_scales,
            cluster=cluster,
            offset=offset,
            event=event,
            baseline_breaks=baseline_breaks,
        )
    if offset is not None or event is not None or baseline_breaks is not None:
        raise ValueError(
            "offset is only for counts; event and baseline_breaks are only for survival"
        )
    start = time.perf_counter()
    draws, warmup, chains, thin = (
        _count(draws, "draws", 2),
        _count(warmup, "warmup", 0),
        _count(chains, "chains", 1),
        _count(thin, "thin", 1),
    )
    if seed is not None:
        seed = _count(seed, "seed", 0)
    if selection not in ("none", "component", "group"):
        raise ValueError("selection must be 'none', 'component', or 'group'")
    if selection == "group" and groups is None:
        raise ValueError("group selection requires groups")
    if selection != "group" and groups is not None:
        raise ValueError("groups may only be supplied with group selection")
    if prior is None:
        prior = Prior()
    if isinstance(prior, dict):
        prior = Prior(**prior)
    if not isinstance(prior, Prior):
        raise ValueError("prior must be a Prior, dictionary, or None")
    clustered = cluster is not None
    scales = {"r": 0.3, "lambda": 0.3}
    if clustered:
        scales["cluster_lambda"] = 0.3
    if proposal_scales is not None:
        if not isinstance(proposal_scales, dict) or set(proposal_scales) - set(scales):
            raise ValueError(f"proposal_scales accepts only {list(scales)}")
        scales.update(proposal_scales)
    scales = {key: _positive(value, f"proposal_scales[{key}]") for key, value in scales.items()}
    data = prepare_data(
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
    )
    n, m = data["Z"].shape
    p = data["X"].shape[1]
    fixed = None
    if r_fixed is not None:
        fixed = np.asarray(r_fixed, dtype=float).copy()
        if fixed.shape != (m,) or not np.isfinite(fixed).all() or (fixed < 0).any():
            raise ValueError(
                "r_fixed must be a finite nonnegative vector with one entry per exposure"
            )
        if selection != "none":
            raise ValueError("r_fixed disallows variable selection")
    # A flat beta prior is improper for separated probit designs. Check the
    # linear adjustment design; the Gaussian-process prior is proper.
    if family == "binomial" and p:
        _check_probit_design(data["y"], data["X"])
    # Conservative allocation guards for the exact dense implementation.
    if 8 * n * n * (9 if clustered else 8) > 512 * 1024**2:
        raise ValueError(
            "dense covariance working storage exceeds the 512 MiB safety budget; use fewer observations"
        )
    saved_bytes = (
        chains * draws * (8 * (p + m + 3 + int(clustered) + (n if family == "binomial" else 0)) + m)
    )
    if saved_bytes > 1024**3:
        raise ValueError(
            "saved posterior arrays exceed the 1 GiB safety budget; reduce draws/chains or fit size"
        )
    shape = (chains, draws)
    samples = dict(
        beta=np.empty(shape + (p,)),
        r=np.empty(shape + (m,)),
        lambda_=np.empty(shape),
        sigma2=np.empty(shape),
        delta=np.empty(shape + (m,), dtype=bool),
        pi=np.empty(shape),
    )
    samples["lambda"] = samples.pop("lambda_")
    if clustered:
        samples["lambda_cluster"] = np.empty(shape)
    if family == "binomial":
        samples["ystar"] = np.empty(shape + (n,))
    selection_units = (
        [np.array([j]) for j in range(m)]
        if selection == "component"
        else [
            np.flatnonzero(np.asarray(data["groups"]) == label)
            for label in dict.fromkeys(data["groups"])
        ]
        if selection == "group"
        else []
    )
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(chains)
    chain_info = []
    for chain_index, child in enumerate(children):
        rng = np.random.default_rng(child)
        chain_initial = initial
        lambda_cluster = None
        if isinstance(initial, dict) and "lambda_cluster" in initial:
            if not clustered:
                raise ValueError("initial lambda_cluster requires cluster labels")
            chain_initial = initial.copy()
            lambda_cluster = _positive(
                chain_initial.pop("lambda_cluster"), "initial lambda_cluster"
            )
        if clustered and lambda_cluster is None:
            lambda_cluster = prior.cluster_lambda_mean
        beta, r, lambda_, sigma2, delta, pi, latent = _initial(
            chain_initial, data, prior, family, selection, fixed, rng
        )
        _, _, lower = covariance_components(
            data["Z"],
            r,
            lambda_,
            cluster_codes=data["cluster_codes"],
            lambda_cluster=lambda_cluster,
        )
        count_keys = (
            ("lambda", "r", "selection", "cluster_lambda")
            if clustered
            else ("lambda", "r", "selection")
        )
        counts = {
            key: {"accepted": 0, "proposed": 0, "warmup_accepted": 0, "warmup_proposed": 0}
            for key in count_keys
        }
        saved = 0
        for iteration in range(warmup + draws * thin):
            is_warmup = iteration < warmup
            if family == "binomial":
                latent = _latent_sweep(data["y"], latent, data["X"] @ beta, lower, rng)
                continuous = latent
            else:
                continuous = data["y"]
            if p:
                solved_x = cho_solve((lower, True), data["X"], check_finite=False)
                precision_beta = data["X"].T @ solved_x
                beta_lower = np.linalg.cholesky((precision_beta + precision_beta.T) / 2)
                beta_mean = cho_solve(
                    (beta_lower, True), solved_x.T @ continuous, check_finite=False
                )
                beta = beta_mean + math.sqrt(sigma2) * solve_triangular(
                    beta_lower.T, rng.normal(size=p), lower=False, check_finite=False
                )
            residual = continuous - data["X"] @ beta
            if family == "gaussian":
                white = solve_triangular(lower, residual, lower=True, check_finite=False)
                rate = prior.sigma2_rate + float(white @ white) / 2
                sigma2 = 1 / rng.gamma(prior.sigma2_shape + n / 2, 1 / rate)
                if not np.isfinite(sigma2) or sigma2 <= 0:
                    raise FloatingPointError("sigma2 update overflowed; rescale the outcome/prior")
            current_ll = _log_likelihood(lower, residual, sigma2)

            def attempt(
                candidate_r, candidate_lambda, log_adjustment, kind, candidate_cluster=None
            ):
                nonlocal r, lambda_, lambda_cluster, lower, current_ll
                counter = counts[kind]
                counter["warmup_proposed" if is_warmup else "proposed"] += 1
                if not np.isfinite(log_adjustment):
                    return False
                try:
                    candidate_cluster = (
                        lambda_cluster if candidate_cluster is None else candidate_cluster
                    )
                    _, _, candidate_lower = covariance_components(
                        data["Z"],
                        candidate_r,
                        candidate_lambda,
                        cluster_codes=data["cluster_codes"],
                        lambda_cluster=candidate_cluster,
                    )
                    candidate_ll = _log_likelihood(candidate_lower, residual, sigma2)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    return False
                ratio = candidate_ll - current_ll + log_adjustment
                if np.isnan(ratio):
                    return False
                uniform = rng.random()
                if uniform == 0 or math.log(uniform) < min(0.0, ratio):
                    r, lambda_, lower, current_ll = (
                        candidate_r.copy(),
                        candidate_lambda,
                        candidate_lower,
                        candidate_ll,
                    )
                    lambda_cluster = candidate_cluster
                    counter["warmup_accepted" if is_warmup else "accepted"] += 1
                    return True
                return False

            log_lambda = math.log(lambda_) + rng.normal(scale=scales["lambda"])
            with np.errstate(over="ignore", under="ignore"):
                candidate = float(np.exp(log_lambda))
            if np.isfinite(candidate) and candidate > 0:
                adjustment = (
                    gamma_logpdf(candidate, prior.lambda_mean, prior.lambda_sd)
                    - gamma_logpdf(lambda_, prior.lambda_mean, prior.lambda_sd)
                    + log_lambda
                    - math.log(lambda_)
                )
                attempt(r, candidate, adjustment, "lambda")
            else:
                attempt(r, lambda_, -np.inf, "lambda")
            if clustered:
                log_cluster = math.log(lambda_cluster) + rng.normal(scale=scales["cluster_lambda"])
                with np.errstate(over="ignore", under="ignore"):
                    candidate_cluster = float(np.exp(log_cluster))
                if np.isfinite(candidate_cluster) and candidate_cluster > 0:
                    adjustment = (
                        gamma_logpdf(
                            candidate_cluster, prior.cluster_lambda_mean, prior.cluster_lambda_sd
                        )
                        - gamma_logpdf(
                            lambda_cluster, prior.cluster_lambda_mean, prior.cluster_lambda_sd
                        )
                        + log_cluster
                        - math.log(lambda_cluster)
                    )
                    attempt(r, lambda_, adjustment, "cluster_lambda", candidate_cluster)
                else:
                    attempt(r, lambda_, -np.inf, "cluster_lambda")
            if selection != "none":
                for unit in selection_units:
                    proposed_delta = delta.copy()
                    candidate_r = r.copy()
                    proposed_delta[unit] = False
                    candidate_r[unit] = 0
                    if rng.random() < pi:
                        member = int(rng.choice(unit))
                        proposed_delta[member] = True
                        candidate_r[member] = prior.sample_r(rng)
                    # Independence proposal is the exact conditional prior:
                    # Bernoulli(pi), uniform member, then relevance prior. Its
                    # forward/reverse densities cancel the entire prior ratio.
                    if attempt(candidate_r, lambda_, 0.0, "selection"):
                        delta = proposed_delta
                included = sum(bool(delta[unit].any()) for unit in selection_units)
                pi = float(
                    rng.beta(
                        prior.inclusion_alpha + included,
                        prior.inclusion_beta + len(selection_units) - included,
                    )
                )
                if not 0 < pi < 1:
                    raise FloatingPointError(
                        "inclusion update hit a floating-point boundary; use less extreme Beta hyperparameters"
                    )
            if fixed is None:
                for j in np.flatnonzero(delta):
                    log_r = math.log(r[j]) + rng.normal(scale=scales["r"])
                    candidate_r = r.copy()
                    with np.errstate(over="ignore", under="ignore"):
                        proposed_r = float(np.exp(log_r))
                    if np.isfinite(proposed_r) and proposed_r > 0:
                        candidate_r[j] = proposed_r
                        adjustment = (
                            prior.log_r(candidate_r[j]) - prior.log_r(r[j]) + log_r - math.log(r[j])
                        )
                        attempt(candidate_r, lambda_, adjustment, "r")
                    else:
                        attempt(candidate_r, lambda_, -np.inf, "r")
            if not is_warmup and (iteration - warmup + 1) % thin == 0:
                for key, value in (
                    ("beta", beta),
                    ("r", r),
                    ("lambda", lambda_),
                    ("sigma2", sigma2),
                    ("delta", delta),
                    ("pi", pi),
                ):
                    samples[key][chain_index, saved] = value
                if clustered:
                    samples["lambda_cluster"][chain_index, saved] = lambda_cluster
                if family == "binomial":
                    samples["ystar"][chain_index, saved] = latent
                saved += 1
        for counter in counts.values():
            counter["rate"] = (
                counter["accepted"] / counter["proposed"] if counter["proposed"] else None
            )
            counter["warmup_rate"] = (
                counter["warmup_accepted"] / counter["warmup_proposed"]
                if counter["warmup_proposed"]
                else None
            )
        chain_info.append(
            {"chain": chain_index, "acceptance": counts, "seed_spawn_key": list(child.spawn_key)}
        )
    settings = dict(
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
        proposal_scales=scales,
        r_fixed=None if fixed is None else fixed.tolist(),
        clustered=clustered,
    )
    sampling_info = dict(
        runtime_seconds=time.perf_counter() - start,
        chains=chain_info,
        algorithm="dense marginal MH/Gibbs",
        proposal_adaptation=False,
        seed_entropy=int(seed_sequence.entropy),
    )
    from .result import BKMRFit

    return BKMRFit(data, samples, settings, sampling_info)
