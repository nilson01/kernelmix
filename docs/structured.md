# Structured models and Gaussian process approximations

The [model guide](methods.md) introduces the exposure function and outcome distribution, following [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058) and the [2018 extension](https://doi.org/10.1186/s12940-018-0413-y). A Gaussian process (GP) prior relates exposure-function values at different exposure profiles.

This guide explains how to change that kernel, reduce the cost of its calculations, model dependence between observations, or describe more detailed event-time records. These options use the `StructuredFit` result class and the separate priors defined below.

Run the blocks in order. Eight observations and four warmup/saved draws per chain illustrate the interface. For inference, use longer chains, sampling diagnostics, prior-sensitivity checks and exposure profiles supported by the data.

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

Supply exposures and knots in original units. Prediction reuses training
standardization. Adjustment and random-slope inputs exclude the added intercept.

## Choose an exposure kernel

The kernel relates exposure profiles (rows of exposure values). For profiles $\mathbf z$ and $\mathbf z'$ with $M$ exposures and relevance parameters $r_m$, define

$$q(\mathbf z,\mathbf z')=\sum_{m=1}^{M}r_m(z_m-z'_m)^2,\qquad r_m\geq0.$$

The profiles use the fitted standardization when enabled. Setting $r_m=0$ removes exposure $m$ from the kernel.

| `kernel` | Kernel at weighted squared distance $q$ |
| --- | --- |
| `"gaussian"` | $\exp(-q)$ |
| `"matern32"` | $(1+\sqrt{6q})\exp(-\sqrt{6q})$ |
| `"rational_quadratic"` | $(1+q/\alpha)^{-\alpha}$ |

The fixed $\alpha>0$ (`kernel_alpha`) defaults to one; changing it requires
`kernel="rational_quadratic"`. As $\alpha$ increases, this kernel approaches the
Gaussian kernel. These definitions follow [GPML, Chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf)
with KernelMix relevance parameters.

```python
matern = km.fit(y, Z, X, kernel="matern32", **demo)
rq = km.fit(y, Z, X, kernel="rational_quadratic", kernel_alpha=1.5, **demo)
curve = km.univariate_response(matern, variable=0, points=5, seed=48)
```

Kernel choice also affects the fitting route and its priors. For exposure-function values $\mathbf h$ at observed profiles, structured fits store `lambda` as the direct GP amplitude $\tau$ in $\mathbf h\sim N(\mathbf0,\tau\mathbf K)$, including Gaussian fits. Default Gaussian fits store $\tau/\sigma^2$. Compare priors on these stated scales. Exposure contrasts remain conditional associations, requiring additional assumptions for causal interpretation.

### Structured priors

| Quantity | Default |
| --- | --- |
| Adjustment coefficients, including the intercept | Independent $N(0,25)$; SD set by `prior.beta_sd` |
| Direct GP amplitude $\tau$ | Gamma(shape 4, rate 4) |
| Gaussian residual variance $\sigma^2$ | Inverse-gamma(shape 2, rate 1) |
| Random-coefficient covariance | Defined in the [random-effects section](#model-random-intercepts-and-slopes) |

A `prior` dictionary changes only the supplied fields; a `Prior` object supplies every field. Check `fit.settings["prior"]` when comparing models. The
`cluster_lambda_mean` and `cluster_lambda_sd` fields apply to default random
intercepts, not structured random coefficients.

## Reduce GP matrix size

After choosing how the kernel relates exposure profiles, choose how to represent the GP in computation. `"dense"` uses the full kernel. `"pp"` represents $h$ through values at a smaller set of exposure profiles (knots). `"rff"` uses fixed sine/cosine features to approximate the kernel.

```python
pp = km.fit(y, Z, X, kernel="matern32", approximation="pp",
            n_components=4, **demo)
rff = km.fit(y, Z, X, kernel="matern32", approximation="rff",
             n_components=8, **demo)
pp_prediction = pp.predict(Z[:2], X[:2], seed=48)
```

For the predictive process (PP), `n_components` caps the number of automatically chosen knots, which are spread across observed exposure profiles; explicit knots can instead
be supplied as `knots=Z[[0, 3, 5, 7]]`. Its covariance is $\tau\mathbf K_{ZU}\mathbf K_{UU}^{+}\mathbf K_{UZ}$, where $U$ denotes the knots and $+$ is the numerical pseudoinverse, an inverse restricted to matrix directions distinguishable from zero. This representation omits the full GP's remaining variation after its knot values are known. Uncertainty can therefore be much smaller away from knots; this is the projected model described by [Banerjee and colleagues](https://academic.oup.com/jrsssb/article-pdf/70/4/825/49796141/jrsssb_70_4_825.pdf), not an exact dense-GP uncertainty calculation.

For random Fourier features (RFF), `n_components` counts frequency pairs, giving twice as many features. Paired sine/cosine features follow [Rahimi and Recht](https://papers.nips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf). Frequencies are Gaussian for the Gaussian kernel, multivariate Student-t with three degrees of freedom for Matérn, and a Gaussian/Gamma scale mixture for RQ. Each mixture scale is shared across all coordinates of one frequency vector. Base draws stay fixed throughout sampling and are saved with the fit. Inference treats this saved set of features as fixed. It does not include uncertainty from choosing a different feature set, and a fixed feature count does not establish approximation accuracy.

During fitting, PP and RFF store matrices with one row per observation and one column per feature, together with square matrices whose size is the feature count. Joint predictions and exposure contrasts still return dense matrices over the requested prediction rows; keep these requests small or use suitable batches. Large correlated Gaussian residual blocks can still require dense calculations. Dense/PP eigensystems treat eigenvalues at most `100 * dimension * machine_epsilon * largest_eigenvalue` as zero for numerical calculations. Eigenvalues measure variation along matrix directions; this threshold removes directions too small to distinguish reliably from zero. Saved dense training values and PP knot values preserve prediction coordinates across reloads.

## Model random intercepts and slopes

For repeated observations, random coefficients describe variation between clusters beyond the shared exposure function. Supply `cluster` labels to identify the observation groups and `random_slopes` to add slopes after the random intercept. All five outcome families support these coefficients. The `groups` argument instead groups exposures for selection.

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

`random_effects()` returns joint coefficient draws; `predict` combines them with the new slope design. A fixed covariance must be positive definite (every nonzero linear combination has positive variance) and match `[intercept, slopes]`. Otherwise, the full covariance is estimated.

The random-coefficient prior separates standard deviations and correlations. Each SD is `random_scale * Phi(raw)`, with standard-normal `raw` and cumulative distribution function `Phi`, giving a uniform SD between zero and `random_scale` (default two). Correlations come from a normalized lower-triangular matrix with normal off-diagonal entries. This proper prior is neither LKJ nor uniform over correlation matrices.

Prediction uses saved coefficients for known clusters. Each distinct new label receives one coefficient vector per draw, shared across its rows. Omitting labels sets random effects to zero. With a nonlinear link, this differs from averaging predictions over new clusters. Supply `random_slopes_new` for new slope rows. Redundant design columns are rejected; few clusters or little slope variation can still make covariance estimates sensitive to priors.

Count/probit inputs retain their usual meanings. For example, an NB2 model can use the same slope workflow:

```python
counts = np.array([0, 1, 2, 1, 0, 2, 1, 3])
nb = km.fit(counts, Z, X, family="negative_binomial", cluster=cluster,
            random_slopes=slope, approximation="rff", n_components=8, **demo)
count_means = nb.predict(Z[:2], X[:2], seed=50)
```

For counts, `offset` still supplies the logarithm of observation time or exposure amount. Response predictions are count means or, with `kind="observation"`, sampled counts. Response-scale exposure contrasts use zero offset and zero random effects.

## Separate Gaussian variance and correlation

Random coefficients describe variation shared within a cluster. A Gaussian residual model describes the remaining variation and can allow residual variances to differ between observations or residuals to be correlated within a cluster. With `variance_design=V`,
$\operatorname{Var}(\epsilon_i\mid\sigma^2,\boldsymbol\gamma)=\sigma^2\exp(\mathbf v_i^\mathsf T\boldsymbol\gamma)$,
where $\mathbf v_i$ is row $i$ of $V$ and $\boldsymbol\gamma\sim N(\mathbf0,\mathbf I)$. No intercept is added. Columns must be linearly independent, and no combination may give a nonzero constant column: `sigma2` already sets the baseline variance. This parameterization is distinct from [nlme's variance function](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/varExp.html).

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

Residual correlation requires Gaussian outcomes and clusters. Within each cluster,
AR1 times must be distinct integers and exponential times distinct finite values.
AR1 uses `rho**gap`, with `-1 < rho < 1`; exponential uses `exp(-gap/range)`,
with positive range. See nlme's [AR1 definition](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corAR1.html) and [exponential definition](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corExp.html). A supplied `correlation_parameter` is fixed. Otherwise, AR1 uses `rho=tanh(raw)` and exponential uses `range=exp(raw)`, each with a standard-normal prior on `raw`.

For known clusters, new residuals may be correlated with training residuals. Mean prediction uses that relationship to predict the new residual component, and observation prediction includes uncertainty in it. Supply original
`cluster_new` labels to condition training-row predictions; their times and slope
rows are reused. Unseen or omitted labels have no training residual
cross-correlation. `predict_h` and exposure contrasts do not use training residuals to predict that component.

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

With fixed correlation and no custom cross matrix, new residuals are independent of training residuals. The combined training/new correlation must be positive semidefinite: no linear combination may have negative variance. Residual variance predictions exclude uncertainty in the GP and random coefficients. Random effects and long-range residual correlation can be hard to distinguish with sparse repeated measurements.

## Encode delayed entry and competing events

For survival data, the likelihood must account for when follow-up starts and what is known about the event. Delayed entry starts follow-up after time zero and includes only subjects event-free until entry. Competing causes identify different first-event types. `y` records an exact event time, right-censoring time or interval lower bound; `event=0` means right censoring, and `1..n_causes` identifies a cause. Only the first terminal event is modeled, excluding recurrent events and general state transitions. Baseline intervals are right closed: an event exactly at a cut belongs to the interval on its left.

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

For cause $k$, the hazard is $\lambda_k(t\mid\mathbf z_i,\mathbf x_i)=\lambda_{0k}(t)\exp\{h_k(\mathbf z_i)+\mathbf x_i^\mathsf T\boldsymbol\beta_k\}$, with random-coefficient terms added when supplied. Cause-specific functions and parameters have independent priors conditional on
shared exposure relevance and selection. Posterior dependence can remain through
shared parameters and interval-censored likelihoods.

For competing risks, survival means no event of any cause; cumulative incidence (CIF) gives the probability of a specified cause by a time. Joint draws satisfy survival plus the sum of CIFs equals one; see the [R survival guide](https://cran.r-project.org/web/packages/survival/vignettes/compete.pdf).

Survival response-scale contrasts subtract relative hazards; they are not hazard ratios or survival-probability differences. To select results, `cif.for_cause(2)` extracts one CIF, whereas a fit's `for_cause(2)` selects its exposure function, relative-hazard predictions and contrasts. `km.plot_survival(cause_two)` plots a CIF with pointwise intervals when Matplotlib is installed.

Prediction `entry` describes subjects known to be event-free through that time; before it, survival is one and CIFs are zero. Static-profile prediction extends the final baseline hazard beyond follow-up. Baseline cuts are held fixed; their selection uncertainty is excluded. An intercept and baseline hazard can compensate for each other. Use `intercept=False` unless separate, prior-dependent estimates are intended.

Delayed-entry fits describe subjects included after surviving to their recorded entry times, at their recorded predictor values. They do not recover risk among people who were never observed before entry, or correct for how survival to entry changes who is included. New-cluster predictions use this same model for included subjects.

## Record left or interval censoring

When an event time is uncertain, specify a finite interval with `upper > y`. Left censoring uses lower bound and entry zero. Exact events use `upper == y` or omit `upper`; right censoring uses `event=0`, optionally with infinite upper bound. Unknown cause `event=-1` requires a finite interval; specify `n_causes` if all event causes are unknown. These are KernelMix codes, not [R Surv status codes](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html).

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

The likelihood assumes that censoring and inspection times provide no further information about event risk after accounting for modeled information. It also assumes that missing cause labels are ignorable after that information is considered. Summing over unknown causes does not model a recording process that depends on the cause itself. An interval row assumes its supplied predictors apply throughout the uncertain event interval.

## Supply known covariate paths

When predictors change over time, use repeated `subject` labels to identify successive `(entry, y]` intervals for the same subject. Rows must be chronological, contiguous and nonoverlapping; only the last may have an event or finite censoring interval. A later row continues the same subject's history rather than representing entry into the cohort. `missing="drop"` removes a whole subject history if required values are missing, preserving the record structure but not correcting missing-data bias.

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

Paths must cover requested times; the final row is never extended.
`kind="hazard"` returns cause-specific hazards. Output follows first occurrence
of `subject_new`. Repeated cluster labels share a random vector across intervals.

Predictors must be known before their risk interval. Do not copy event-time or
later measurements backward. Paths are neither interpolated nor forecast. Draws
are joint across times and causes; credible intervals are pointwise.

## Set starting values

Structured `initial` dictionaries accept `beta`, `gp_weight`, `r`, `delta`, `pi`, `lambda` and `sigma2`; NB2 adds `dispersion` and survival adds `baseline_hazard`. Omitted values receive defaults. With $K$ causes (one for nonsurvival models; this dimension label is distinct from the kernel matrix $\mathbf K$), $p$ adjustment coefficients including the intercept, and $b$ retained basis coordinates:

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

Random coefficients, covariance coordinates, variance coefficients and inferred residual-correlation coordinates start at their internal defaults. Those internal sampling variables cannot be set through `initial`. Dense basis dimension is the retained observation count, PP dimension is the knot count, and RFF dimension is twice `n_components`; numerical null columns are retained. Omit `gp_weight` to use zero starting coordinates.

Saved structured arrays have leading `(chain, draw)` axes. They include cause-indexed `beta`, `lambda` and `gp_weight`, shared `r`, `delta` and `pi`, and Gaussian `sigma2`. Dense/PP fits retain coordinate-independent `gp_anchor` values; RFF fits retain their frozen feature map. Applicable models also save `random_coeff`, `random_covariance`, `gamma`, `correlation_parameter`, `dispersion` and `baseline_hazard`. Use result methods for prediction rather than interpreting internal coordinates as exposure effects.
