# Structured models and GP approximations

Use these options when the analysis needs a different exposure kernel, a finite GP basis, random slopes, Gaussian residual structure, or extended survival records. `km.fit` returns a `StructuredFit` when one of these options is selected. The ordinary defaults retain the original fitting route. The underlying exposure model follows [Bobb and colleagues](https://doi.org/10.1093/biostatistics/kxu058), with the [software extension](https://doi.org/10.1186/s12940-018-0413-y); the additional structures below have their own stated definitions.

The examples below share eight fixed observations. Run the blocks in order. Four saved draws and four warmup iterations per chain show the workflow only; they cannot support inference. Longer runs, sampling diagnostics, prior sensitivity and checks of data support are needed for an analysis.

```python
import numpy as np
import kernelmix as km

x = np.linspace(-1.0, 1.0, 8)
visit = np.tile(np.arange(4), 2)
cluster = np.array(["a"] * 4 + ["b"] * 4)
Z = np.column_stack((2 + x, 5 + np.sin(np.arange(8))))
X = np.cos(np.arange(8))[:, None]
y = 0.3 + x + 0.2 * np.sin(2 * np.arange(8))
slope = x[:, None]
V = (x - x.mean())[:, None]
demo = dict(draws=4, warmup=4, chains=2, seed=47)
```

All exposure inputs and explicit knots use original exposure units. Training standardization is stored and reused. Adjustment and random-slope inputs exclude the automatically added intercept.

## Choose an exposure kernel

For exposure profiles $\mathbf z$ and $\mathbf z'$, define

$$q(\mathbf z,\mathbf z')=\sum_{m=1}^{M}r_m(z_m-z'_m)^2,\qquad r_m\geq0.$$

The profiles use the fitted standardization when enabled. Setting $r_m=0$ removes exposure $m$ from the kernel.

| `kernel` | Kernel at weighted squared distance $q$ |
| --- | --- |
| `"gaussian"` | $\exp(-q)$ |
| `"matern32"` | $(1+\sqrt{6q})\exp(-\sqrt{6q})$ |
| `"rational_quadratic"` | $(1+q/\alpha)^{-\alpha}$ |

For the rational-quadratic kernel, $\alpha$ (`kernel_alpha`) is fixed, positive and defaults to one. A nondefault value requires `kernel="rational_quadratic"`. Increasing it makes the rational-quadratic kernel approach the Gaussian kernel. These are the Matérn and rational-quadratic definitions in [GPML, Chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf), expressed using KernelMix's relevance weights.

```python
matern = km.fit(y, Z, X, kernel="matern32", **demo)
rq = km.fit(y, Z, X, kernel="rational_quadratic", kernel_alpha=1.5, **demo)
curve = km.univariate_response(matern, variable=0, points=5, seed=48)
```

Structured fits store `lambda` as the **direct GP covariance amplitude**, corresponding to Bobb's $\tau$: $\mathbf h\sim N(\mathbf0,\tau\mathbf K)$. This also applies to structured Gaussian fits. The legacy Gaussian route instead stores the ratio $\tau/\sigma^2$. Selecting a structured model therefore changes its prior specification as well as its covariance options. Exposure contrasts describe conditional associations; causal interpretation requires additional assumptions.

Structured adjustment coefficients, including an intercept when present, have independent normal priors with standard deviation `prior.beta_sd`, default five. Default GP amplitudes have Gamma shape four and rate four. Structured Gaussian residual variance defaults to inverse-gamma shape two and rate one, meaning its reciprocal has that Gamma distribution. A `prior` dictionary overrides only supplied fields on these structured defaults; a `Prior` object supplies all its fields directly. Inspect `fit.settings["prior"]` when comparing models. The legacy `cluster_lambda_mean` and `cluster_lambda_sd` fields do not govern structured random coefficients; their covariance prior is defined below.

## Reduce GP matrix size

`approximation="dense"` uses the full kernel. `"pp"` uses a predictive process at a fixed set of knots. `"rff"` uses frozen random Fourier features.

```python
pp = km.fit(y, Z, X, kernel="matern32", approximation="pp",
            n_components=4, **demo)
rff = km.fit(y, Z, X, kernel="matern32", approximation="rff",
             n_components=8, **demo)
pp_prediction = pp.predict(Z[:2], X[:2], seed=48)
```

For PP, `n_components` is the maximum number of automatic, deterministic max-min knots. Alternatively, supply `knots=Z[[0, 3, 5, 7]]`. Its covariance is $\tau\mathbf K_{ZU}\mathbf K_{UU}^{+}\mathbf K_{UZ}$, where $U$ denotes the knots and $+$ is the numerical pseudoinverse. It omits the parent GP's conditional residual variance. Uncertainty can therefore be much smaller away from knots; this is the projected model described by [Banerjee and colleagues](https://academic.oup.com/jrsssb/article-pdf/70/4/825/49796141/jrsssb_70_4_825.pdf), not an exact dense-GP uncertainty calculation.

For RFF, `n_components` counts frequency pairs, giving twice as many features. Paired sine/cosine features follow [Rahimi and Recht](https://papers.nips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf). Frequencies are Gaussian for the Gaussian kernel, multivariate Student-t with three degrees of freedom for Matérn, and a Gaussian/Gamma scale mixture for RQ. Each mixture scale is shared across all coordinates of one frequency vector. Base draws stay fixed throughout sampling and are saved with the fit. Inference is conditional on that realized feature map; a fixed feature count does not establish approximation accuracy.

During fitting, PP/RFF use observation-by-feature and feature-by-feature matrices. Joint predictions and exposure contrasts still return dense matrices over the requested prediction rows; keep these requests small or use suitable batches. Large correlated Gaussian residual blocks can still require dense calculations. Dense/PP eigensystems treat eigenvalues at most `100 * dimension * machine_epsilon * largest_eigenvalue` as numerical null modes. Saved dense training values and PP knot values preserve prediction coordinates across reloads.

## Model random intercepts and slopes

`cluster` identifies observation groups; exposure-selection `groups` has a separate meaning. `random_slopes` adds columns after the random intercept. All five outcome families support these coefficients.

```python
slopes = km.fit(y, Z, X, cluster=cluster, random_slopes=slope, **demo)
coefficients = slopes.random_effects(["a", "new"], seed=49)
intercepts = slopes.random_intercepts(["a", "new"], seed=49)
prediction = slopes.predict(
    Z[:2], X[:2], cluster_new=["a", "new"], random_slopes_new=slope[:2], seed=49,
)
fixed_slopes = km.fit(
    y, Z, X, cluster=cluster, random_slopes=slope,
    random_covariance=[[0.5, 0.1], [0.1, 0.3]], **demo,
)
```

`random_effects()` returns joint coefficient draws; `predict` combines them with the new slope design. A fixed covariance must be positive definite and match `[intercept, slopes]`. Otherwise, the full covariance is estimated. Each standard deviation is `random_scale * Phi(raw)`, where `raw` is standard normal and `Phi` is the standard normal cumulative distribution function. Thus SDs are uniform between zero and `random_scale`, default two. A normalized lower-triangular factor with normal off-diagonal coordinates supplies a proper correlation prior. It is neither an LKJ prior nor a uniform prior on correlation matrices.

Known labels use saved coefficients. Each distinct new label receives one fresh vector per draw, shared across its rows. Omitting `cluster_new` sets random effects to zero; on nonlinear links this differs from averaging over new clusters. Supply `random_slopes_new` when predicting their contribution at new rows. Redundant random-design columns are rejected, but few clusters or little slope variation can still leave covariance estimates strongly dependent on priors.

Count/probit inputs retain their usual meanings. For example, an NB2 model can use the same slope workflow:

```python
counts = np.array([0, 1, 2, 1, 0, 2, 1, 3])
nb = km.fit(counts, Z, X, family="negative_binomial", cluster=cluster,
            random_slopes=slope, approximation="rff", n_components=8, **demo)
count_means = nb.predict(Z[:2], X[:2], seed=50)
```

Count `offset` remains a log observation-exposure offset. Response predictions are count means or, with `kind="observation"`, sampled counts. Response-scale exposure contrasts use zero offset and zero random effects. Survival contrasts describe differences in relative hazards; they are not hazard ratios or survival-probability differences.

## Separate Gaussian variance and correlation

With `variance_design=V`, $\operatorname{Var}(\epsilon_i\mid\sigma^2,\boldsymbol\gamma)=\sigma^2\exp(\mathbf v_i^\mathsf T\boldsymbol\gamma)$, where $\mathbf v_i$ is row $i$ of $V$ and $\boldsymbol\gamma\sim N(\mathbf0,\mathbf I)$. No intercept is added to `V`. Its columns must be independent and their span must exclude a constant, because `sigma2` already controls the baseline variance. This is an exponential variance model with the package's own parameterization; compare [nlme's variance-function definition](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/varExp.html).

```python
ar = km.fit(
    y, Z, X, cluster=cluster, variance_design=V,
    correlation="ar1", observation_time=visit, correlation_parameter=0.4, **demo,
)
future_rows = np.array([3, 7])
future_mean = ar.predict(
    Z[future_rows], X[future_rows], cluster_new=["a", "b"],
    observation_time_new=[4, 4], variance_design_new=V[future_rows], seed=51,
)
residual_variance = ar.predict_residual_variance(
    Z[future_rows], variance_design_new=V[future_rows],
)
exponential = km.fit(
    y, Z, X, cluster=cluster, correlation="exponential",
    observation_time=visit / 2, **demo,
)
```

Residual correlation applies only to Gaussian outcomes and requires clusters. AR1 times are distinct integers within each cluster; exponential times are distinct finite values. AR1 correlation at a time gap is `rho**gap`, with `-1 < rho < 1`; exponential correlation is `exp(-gap/range)`, with positive range. See nlme's [AR1 definition](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corAR1.html) and [exponential definition](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corExp.html). A supplied `correlation_parameter` is fixed. Otherwise, AR1 uses `rho=tanh(raw)` and exponential uses `range=exp(raw)`, each with a standard-normal prior on `raw`.

For a known cluster, correlated prediction conditions on training residuals after subtracting its fitted structural predictor. Mean prediction includes the conditional residual mean; observation prediction also includes conditional residual uncertainty. An unseen cluster has no training residual cross-correlation. To condition predictions for training rows on the recorded process, supply their original `cluster_new` labels; training times and slope rows are reused. Omitting labels supplies no training residual cross-correlation. `predict_h` and exposure contrasts retain the exposure/structural target rather than this residual-process target.

A fixed correlation matrix must cover all original rows, have unit diagonal, be positive definite, and be zero between clusters. Supply new-by-training and new-by-new correlations to condition new residuals:

```python
R = np.where(cluster[:, None] == cluster[None, :],
             0.3 ** np.abs(visit[:, None] - visit[None, :]), 0.0)
fixed = km.fit(y, Z, X, cluster=cluster, correlation=R, **demo)
cross = np.array([
    np.where(cluster == label, 0.3 ** (4 - visit), 0.0)
    for label in ("a", "b")
])
fixed_prediction = fixed.predict(
    Z[future_rows], X[future_rows], cluster_new=["a", "b"],
    correlation_cross=cross, correlation_new=np.eye(2), seed=52,
)
```

With fixed correlation and no custom cross matrix, new residuals are independent of training residuals. The combined training/new correlation must be positive semidefinite. Residual variance predictions exclude uncertainty in the GP and random coefficients. Random effects and long-range residual correlation can be hard to distinguish with sparse repeated measurements.

## Encode delayed entry and competing events

For survival, `y` is an exact/censoring time or an interval lower bound. `event=0` means right censoring; positive integers `1..n_causes` identify causes. The first terminal event is modeled; recurrent events and general state transitions are excluded. Baseline intervals are right closed, so an event exactly at a cut belongs to its left interval.

```python
time = np.array([0.8, 1.4, 2.2, 3.2, 1.1, 1.9, 2.8, 3.6])
event = np.array([1, 0, 2, 1, 0, 2, 0, 1])
entry = np.array([0.0, 0.2, 0.5, 0.4, 0.1, 0.3, 0.6, 0.2])
survival = km.fit(
    time, Z, X, family="survival", event=event, entry=entry, n_causes=2,
    baseline_breaks=[1.5, 2.5], intercept=False, **demo,
)
times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
s = survival.predict_survival(Z[:2], X[:2], times=times, entry=0.2, seed=53)
cif = survival.predict_cumulative_incidence(Z[:2], X[:2], times=times, entry=0.2, seed=53)
cause_two = survival.predict_cumulative_incidence(
    Z[:2], X[:2], times=times, cause=2, entry=0.2, seed=53,
)
hazard_two = survival.for_cause(2).predict_survival(
    Z[:2], X[:2], times=times, kind="hazard", entry=0.2, seed=53,
)
cause_contrast = km.overall_effect(survival.for_cause(2), seed=53)
```

For cause $k$, the hazard is $\lambda_k(t\mid\mathbf z_i,\mathbf x_i)=\lambda_{0k}(t)\exp\{h_k(\mathbf z_i)+\mathbf x_i^\mathsf T\boldsymbol\beta_k\}$, with random-coefficient terms added when supplied. The cause-specific exposure functions, adjustment coefficients, amplitudes and baselines are independent conditional on the shared exposure relevance and selection parameters; they need not be independent after those parameters are integrated out.

Survival is the probability of no event of any cause. A cumulative incidence function (CIF) is the probability of a particular cause occurring by a time, accounting for all competing causes. Joint draws satisfy survival plus the sum of CIFs equals one. This distinction is explained in the [R survival competing-risk guide](https://cran.r-project.org/web/packages/survival/vignettes/compete.pdf). `cif` retains all causes; `cif.for_cause(2)` selects one. A fit's `for_cause(2)` view instead selects its exposure function, relative-hazard predictions and contrasts. `km.plot_survival(cause_two)` plots a selected CIF with pointwise intervals when Matplotlib is installed.

Prediction `entry` conditions on survival to that time; before it, survival is one and CIFs are zero. Static-profile prediction extends the final baseline hazard beyond follow-up. Baseline cuts are held fixed; their selection uncertainty is excluded. A survival intercept and baseline share a scale, so `intercept=False` is useful unless their separate prior-dependent roles are intended.

These delayed-entry fits describe the **included entrant cohort**, conditional on its recorded entry and predictor design. GP/random-effect priors do not estimate an inception-population frailty distribution or correct its change under survival-to-entry selection. New-cluster predictions inherit this cohort-conditional model.

## Record left or interval censoring

Supply `upper > y` for a finite event interval. A left-censored record has lower bound zero and entry zero. An exact event uses `upper == y` or omits `upper`; right censoring uses `event=0` and may use infinite upper bound. Unknown cause `event=-1` is allowed only for a finite interval. Set `n_causes` explicitly when all observed interval causes are unknown. These inputs use explicit KernelMix codes rather than the alternative status codes of [R's Surv](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html).

```python
lower = np.array([0.0, 1.4, 1.8, 3.2, 1.1, 1.5, 2.8, 3.0])
upper = np.array([0.8, np.inf, 2.2, 3.2, np.inf, 1.9, np.inf, 3.6])
interval_event = np.array([1, 0, 2, 1, 0, -1, 0, 1])
interval = km.fit(
    lower, Z, X, family="survival", event=interval_event, upper=upper,
    entry=np.zeros(8), subject=np.arange(8), n_causes=2,
    baseline_breaks=[1.5, 2.5], intercept=False, **demo,
)
```

The likelihood assumes noninformative censoring and inspections, and ignorable cause recording conditional on modeled information. Unknown-cause integration does not model a cause-dependent recording mechanism. An interval row assumes its supplied predictors apply throughout the uncertain event interval.

## Supply known covariate paths

Repeated `subject` labels identify successive `(entry, y]` risk intervals. Rows must be chronological, contiguous and nonoverlapping within a subject; only its last row may have an event or a finite censoring interval. Later row starts split an existing history rather than representing new cohort entry. Known subject histories with missing required values are dropped as a whole under `missing="drop"`; this preserves recorded histories but does not remove missing-data bias.

```python
subject = np.repeat(np.arange(4), 2)
start = np.tile([0.0, 1.0], 4)
stop = np.tile([1.0, 3.0], 4)
path_event = np.array([0, 1, 0, 0, 0, 2, 0, 1])
path_fit = km.fit(
    stop, Z, X, family="survival", event=path_event, entry=start,
    subject=subject, n_causes=2, baseline_breaks=[1.5, 2.5],
    intercept=False, **demo,
)
path_inputs = dict(start=start, stop=stop, subject_new=subject,
                   times=[0.0, 1.0, 2.0, 3.0], seed=54)
path_survival = path_fit.predict_survival_path(Z, X, **path_inputs)
path_cif = path_fit.predict_survival_path(
    Z, X, kind="cumulative_incidence", **path_inputs,
)
path_cause_two = path_fit.predict_survival_path(
    Z, X, kind="cumulative_incidence", cause=2, **path_inputs,
)
```

Path prediction integrates only the supplied piecewise-constant predictors. Paths must cover requested times through each subject's last stop; the last row is never extended automatically. Use `kind="hazard"` for evaluated cause-specific hazards. Returned rows follow the first occurrence of `subject_new`. Repeated new cluster labels share their random vector across path intervals when supplied.

Covariates must be known before the risk interval they describe. Measurements observed at an event or later inspection cannot be copied backward to create a past path. These functions neither interpolate missing trajectories nor forecast future covariates. Intervals are evaluated jointly across times and causes; the reported credible intervals are pointwise, not simultaneous bands.

## Set starting values

Structured `initial` dictionaries accept `beta`, `gp_weight`, `r`, `delta`, `pi`, `lambda` and `sigma2`; NB2 adds `dispersion` and survival adds `baseline_hazard`. Omitted values receive defaults. With $K$ causes (one for nonsurvival models), $p$ adjustment coefficients including the intercept, and $b$ retained basis coordinates:

| Key | Required shape and restriction |
| --- | --- |
| `beta` | $(K,p)$; a length-$p$ vector is also accepted when $K=1$ |
| `gp_weight` | $(K,b)$; a length-$b$ vector is also accepted when $K=1$ |
| `r`, `delta` | One value per exposure; relevance and zero/one inclusion must agree |
| `lambda` | Positive scalar or length-$K$ vector of direct amplitudes |
| `sigma2` | Positive scalar; fixed at one outside Gaussian models |
| `pi` | Inclusion probability; one without selection |
| `dispersion` | Positive scalar for NB2 |
| `baseline_hazard` | Positive $(K,J)$ array for $J$ baseline intervals |

Random coefficients, covariance coordinates, variance coefficients and inferred residual-correlation coordinates start at their internal defaults. Their raw sampling coordinates are not an initialization interface. Dense basis dimension is the retained observation count, PP dimension is the knot count, and RFF dimension is twice `n_components`; numerical null columns are retained. Omit `gp_weight` to use zero starting coordinates.

Saved structured arrays have leading `(chain, draw)` axes. They include cause-indexed `beta`, `lambda` and `gp_weight`, shared `r`, `delta` and `pi`, and Gaussian `sigma2`. Dense/PP fits retain coordinate-independent `gp_anchor` values; RFF fits retain their frozen feature map. Applicable models also save `random_coeff`, `random_covariance`, `gamma`, `correlation_parameter`, `dispersion` and `baseline_hazard`. Use result methods for prediction rather than interpreting internal coordinates as exposure effects.
