# Model and interpretation

## Continuous outcomes

Following [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058), we separate the exposure function from linear adjustment for other measured variables:

$$
Y_i=h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta+\epsilon_i,
\qquad \epsilon_i\overset{\mathrm{iid}}{\sim}N(0,\sigma^2).
$$

Here, $Y_i$ is the outcome, $\mathbf z_i$ contains the $M$ exposures,
$\mathbf x_i$ contains adjustment covariates, and $\boldsymbol\beta$ contains
their coefficients. The error $\epsilon_i$ is independent normal noise with variance $\sigma^2$. The function $h$ allows curved exposure associations
and interactions, where the association with one exposure depends on another. An exposure profile is a row of exposure values. A Gaussian process (GP) prior describes how values of $h$ vary together across profiles before using the outcome data. At the observed profiles, this prior is

$$
\mathbf h=(h_1,\ldots,h_n)^{\mathsf T}\sim N(\mathbf0,\tau\mathbf K_{Z,\mathbf r}),
\qquad
K(\mathbf z,\mathbf z';\mathbf r)=
\exp\left\{-\sum_{m=1}^{M}r_m(z_m-z'_m)^2\right\}.
$$

Here $h_i=h(\mathbf z_i)$, $n$ is the observation count, $\tau$ controls prior variation in $h$, and $\mathbf K_{Z,\mathbf r}$ is the matrix of kernel values between observed profiles. The kernel measures similarity between profiles. The relevance parameters $r_m\geq0$ determine how strongly each exposure changes that similarity. An exposure with $r_m=0$ does not affect $h$. In the default marginalized Gaussian sampler,
`lambda` is $\lambda=\tau/\sigma^2$. Integrating out $\mathbf h$ gives
covariance $\sigma^2(\mathbf I+\lambda\mathbf K_{Z,\mathbf r})$. Here $\mathbf I$ is the identity matrix. Integrating out $\mathbf h$ means accounting for its distribution without sampling each of its values during parameter fitting.

The code uses the following names for the paper's quantities. The meaning of `lambda` depends on the fitting route.

| Paper quantity | Meaning | API name |
| --- | --- | --- |
| $\mathbf z_i$, $\mathbf x_i$ | Exposure and adjustment rows | Rows of `Z` and `X` |
| $h$, $\boldsymbol\beta$ | Exposure function and adjustment coefficients | `predict_h`, `beta` |
| $\sigma^2$, $r_m$ | Residual variance and exposure relevance | `sigma2`, `r` |
| $\tau$ | GP covariance amplitude | Default Gaussian/probit: `lambda` stores $\tau/\sigma^2$; structured and default count/survival: it stores $\tau$ directly |

The [structured guide](structured.md) defines its separate priors. Equal numerical values of `lambda` need not describe the same covariance across these routes.

## Binary outcomes

With `family="binomial"`, the model uses a probit link:

$$
\Pr(Y_i=1\mid h,\boldsymbol\beta)=
\Phi\{h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta\}.
$$

Here $\Phi$ is the standard normal cumulative distribution function. The latent normal scale is the underlying normal variable whose threshold determines the binary outcome. Its residual
variance is fixed at one. This extension follows
[Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y).
Probability contrasts depend on both exposure profiles and adjustment covariates.
They are not odds ratios. The default sampler rejects complete or quasi-complete separation: adjustment covariates that separate the outcome classes, possibly with some observations on the separating boundary. With a flat coefficient prior, the posterior can then fail to have finite total probability. Structured fits use a proper normal coefficient prior, a distribution with total probability one.

## Count outcomes

For Poisson and NB2 outcomes, define

$$
\eta_i=h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta+u_{c(i)},
\qquad \mu_i=\exp(\eta_i+o_i).
$$

The known log offset $o_i$ accounts for observation time or exposure amount.
Without clustering, $u_{c(i)}=0$. The Poisson model has conditional mean and
variance $\mu_i$. The NB2 model has

$$
E(Y_i\mid\mu_i,\phi)=\mu_i,\qquad
\mathrm{Var}(Y_i\mid\mu_i,\phi)=\mu_i+\mu_i^2/\phi,
\qquad \phi>0.
$$

`dispersion` stores $\phi$. Larger values approach Poisson variation.
These likelihoods follow the standard
[Poisson and NB2 parameterizations](https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html).
R `bkmr` 0.2.2 does not implement these count likelihoods.

## Survival outcomes

For positive observed time $t_i$ and event indicator $d_i$, the model is

$$
\lambda_i(t)=\lambda_0(t)\exp(\eta_i),\qquad
S_i(t)=\exp\{-\Lambda_0(t)\exp(\eta_i)\}.
$$

A hazard is the instantaneous event rate among subjects still event-free. Here $\lambda_0(t)$ is the baseline hazard, $\Lambda_0(t)$ is its integral, and $S_i(t)$ is the probability of remaining event-free through time $t$.
The linear predictor $\eta_i$ is defined above without a count offset.
The indicator $d_i$ is one for an observed event and zero when follow-up ends first (right censoring). Under independent right censoring, the likelihood contribution is
$\{\lambda_i(t_i)\}^{d_i}S_i(t_i)$.
The model uses constant baseline rates $\lambda_{0j}$ within fixed,
right-closed intervals. If $D_{ij}$ is subject $i$'s time in interval $j$,
$j(i)$ contains its event time, and $d_j$ counts events in interval $j$, then

$$
\ell=\sum_i d_i\{\log\lambda_{0,j(i)}+\eta_i\}
-\sum_i e^{\eta_i}\sum_jD_{ij}\lambda_{0j}.
$$

An event at a cut belongs to the interval on its left. Every event then has
positive time at risk in its event interval. With independent
$\lambda_{0j}\sim\mathrm{Gamma}(a_0,b_0)$ priors, the conditional update is

$$
\lambda_{0j}\mid\cdots\sim
\mathrm{Gamma}\left(a_0+d_j,\ b_0+\sum_iD_{ij}e^{\eta_i}\right).
$$

This is a full piecewise-exponential event-time likelihood, not Cox partial
likelihood. See the [INLA manual, section 3.15](https://inla.r-inla-download.org/r-inla.org/doc/inla-manual/inla-manual.pdf)
for the model form and the [Stan survival guide](https://mc-stan.org/docs/stan-users-guide/survival.html)
for censoring contributions. KernelMix uses its own independent gamma baseline
prior. Once selected, the time cutpoints are held fixed. The model does not include uncertainty from choosing them. Defaults use event-time deciles, so assess sensitivity to their number
and placement. Baseline priors depend on the chosen time unit.

The likelihood above describes the default right-censored model. The
[structured survival interface](structured.md) also supports delayed entry,
competing causes, predictor histories and left/interval censoring. All-censored data are accepted with a warning:
the posterior then depends strongly on the priors. A fitted intercept and the
baseline hazard share a scale. Separate estimates are anchored by their priors;
`intercept=False` avoids adding that redundant coefficient.

## Random intercepts

For the default random-intercept route, with observation-cluster index $c(i)$, independent Gaussian intercepts satisfy
$u_g\sim N(0,\sigma_u^2)$. Exposures in `groups` and observations in `cluster`
serve different purposes. Conditional on $h$, covariates and random intercepts,
observations are independent under the selected outcome likelihood.

For default Gaussian/probit fits, `lambda_cluster` is $\sigma_u^2/\sigma^2$, with probit
$\sigma^2=1$. The marginalized covariance is

$$
\sigma^2\{\mathbf I+\lambda\mathbf K_{Z,\mathbf r}
+\lambda_{u}\mathbf J\},\qquad
J_{ij}=1\{c(i)=c(j)\}.
$$

For count/survival fits there is no residual Gaussian variance parameter:
`lambda` and `lambda_cluster` are the GP and random-intercept variances directly.
The random-intercept model follows the Gaussian/probit extension described by
[Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y).
KernelMix also combines these intercepts with count and survival likelihoods.

Prediction for a known cluster uses its fitted random intercept. Equal unseen labels
share a new Gaussian draw. Omitted labels fix $u=0$; this differs from averaging predictions over the distribution of cluster effects. Random intercepts do not provide arbitrary serial
correlation or cluster-robust inference. The data may provide little information to distinguish GP variation from cluster variation, particularly for constant kernels or clusters with only one observation.

## Priors and exposure selection

Default Gaussian/probit priors follow R `bkmr` 0.2.2. The [structured guide](structured.md) defines its separate priors. A prior describes parameter values before using the outcome data. Gamma distributions below use shape and **rate**. The inverse-gamma density is proportional to
$(\sigma^2)^{-a-1}\exp(-b/\sigma^2)$ for shape $a$ and parameter $b$.

| Quantity | Default |
| --- | --- |
| $\lambda$ and $\lambda_u$, when clustered | Gamma(1, 0.1) |
| $\sigma^2$, continuous outcome | Inverse-gamma(0.001, 0.001) |
| Active $r_m$ | $1/r_m\sim\mathrm{Uniform}(0,100)$ |
| Inclusion probability $\pi$ | Beta(1,1) |
| $\boldsymbol\beta$ | Flat prior |

Count/survival `fit(..., prior=None)` instead uses
$\lambda,\lambda_u\sim\mathrm{Gamma}(4,4)$, independent
$\boldsymbol\beta\sim N(\mathbf0,25\mathbf I)$,
$\phi\sim\mathrm{Gamma}(2,0.1)$ for NB2, and
$\lambda_{0j}\sim\mathrm{Gamma}(0.1,0.1)$ for survival.
At fixed finite predictor values, the resulting prior distributions of the count mean and relative hazard have finite means and second moments. A supplied `Prior` is used as specified. A partial dictionary overlays
the selected family's defaults, for example `prior={"r_distribution": "gamma"}`.
Warnings identify supplied amplitude or dispersion priors for which these moment
guarantees do not hold. Assess prior sensitivity; large finite moments can still
be numerically unstable. `sigma2=1` is a storage convention for count/survival fits and is omitted from
their summaries.

The inverse-uniform relevance prior has an unbounded upper tail. An average from a finite sampling run can be calculated even when the posterior mean of $r_m$ does not exist. Inspect
quantiles and mixing; assess prior sensitivity for the application.
`Prior(r_distribution="gamma")` and `"unif"` provide other positive priors.
The gamma parameters use mean and standard deviation. Uniform bounds apply to
$r_m$ for `"unif"` and to $1/r_m$ for `"invunif"`.

With component selection, $\delta_m\sim\mathrm{Bernoulli}(\pi)$ indicates whether
$r_m>0$. Excluded exposures have $r_m=0$. Group selection uses the groups $S_g$
and indicator $\omega_g$ from Bobb et al. (2015): each active group contains
**one** selected exposure, chosen uniformly within that group. KernelMix does
not offer unequal within-group prior probabilities. A group PIP is its posterior
inclusion probability; a conditional PIP is an exposure's inclusion probability
given that its group is active. A never-selected group has an undefined
conditional PIP. PIPs depend on the model and priors; they do not measure causal
importance.

## Inputs and computation

`fit` adds an intercept and standardizes exposures by default. Standardization
uses the retained training mean and sample standard deviation. Prediction reuses
these values. Pass `intercept=False` or `standardize=False` to disable either.
Do not include a constant adjustment column when adding an intercept.
R `bkmr` does not automatically apply these Python preprocessing defaults.

`missing="raise"` rejects missing data. `missing="drop"` removes rows jointly
across required inputs, including applicable cluster labels, offsets and survival
inputs. For recorded subject histories, it removes the whole affected history;
see [structured survival](structured.md#supply-known-covariate-paths). Retained
original row indices are stored in `result.data["rows"]`. Complete-case fitting does not correct bias
caused by missing data. Covariates must be numeric; categorical encoding is the
caller's responsibility.

A chain is one sampling run through the posterior distribution, which combines the data and priors. Chains run sequentially with separate random streams. `draws` counts saved draws
per chain after `warmup`; `thin` controls the spacing of saved draws. Proposals
are fixed during sampling. `sampling_info` records acceptance counts and runtime.
The default Gaussian sampler integrates out $h$ during parameter sampling. The default binary sampler also samples underlying normal outcomes constrained to be positive or negative according to the observed binary response. Predictions use those saved normal outcomes.

Default count/survival fits sample a noncentered GP with
[elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html),
log-scale Metropolis updates and conjugate baseline updates. Eigenvalues at most
$100n\epsilon\max(1,\lambda_{\max}(\mathbf K))$, where $\epsilon$ is machine
precision, are treated as zero in numerical calculations. Eigenvalues measure variation along matrix directions; very small ones cannot be distinguished reliably from zero at machine precision. No small constant is added to the kernel diagonal (diagonal jitter).
Prediction uses the same rank rule and the saved latent function. Saved auxiliary white coordinates are internal, standardized sampling variables. Their eigenvector signs can differ across platforms without changing the represented function. Archive validation checks that values lie in the supported matrix directions rather than requiring identical signs.

Structured fits instead sample the GP and other normal-prior coordinates jointly by elliptical slice sampling, with log-scale Metropolis updates for amplitudes, relevance, dispersion and survival baseline rates. Gaussian residual variance has a conditional inverse-gamma update. Their priors and covariance options are defined in the [structured guide](structured.md).

Dense matrix factorization requires roughly cubic work in $n$ per parameter
update: doubling the observation count can multiply this work by about eight. Working-memory checks reject oversized fitting or prediction requests;
they are protective estimates, not a promise about peak process memory.

## Prediction and contrasts

`predict_h` draws $h$ jointly over requested profiles. `predict(kind="mean")`
returns outcome means, binary probabilities, or relative hazards for survival.
`predict_survival` gives survival probabilities or hazards on a time grid.
`predict(kind="observation")` also includes new observation variation for
Gaussian, binary and response-scale count predictions. Survival observation
sampling is not implemented.
The latent binary scale is the underlying normal scale.

A contrast is a difference between fitted values. Contrasts subtract **joint** posterior draws, which contain predictions at all compared profiles together. This preserves how their uncertainty varies together.
`overall_effect` compares all-exposure quantile profiles with a reference
profile. `single_exposure_effects` changes one exposure while holding the others
fixed. `interaction_effects` returns a difference of two single-exposure
contrasts. Curves and surfaces vary one or two exposures; others are held fixed.
Their equal-tailed intervals leave the same posterior probability below and above each interval. They describe uncertainty at individual points, not a simultaneous band for a whole curve. On the response scale, adjustment covariates default to their retained
training means unless `X` supplies a different profile.

A quantile profile combines a percentile chosen separately for each exposure. That combination may be uncommon when exposures are correlated. Check whether observed exposure rows are close to the requested combinations before interpreting contrasts. Causal interpretation requires
additional identification assumptions; fitting a model does not supply them.

## Diagnostics and scope

`diagnostics()` reports rank-normalized/folded split $\widehat R$, bulk effective
sample size and tail effective sample size. R-hat compares sampled distributions across parts of the chains; values near one are desirable. Effective sample size accounts for dependence between draws, in the main part (bulk) or extremes (tails) of the distribution. Large $\widehat R$, small effective
sample sizes, or constant sampled traces need investigation. Fixed parameters
are excluded from these diagnostics. Trace plots and sensitivity checks remain
necessary; favorable diagnostics do not prove convergence.

KernelMix supports Gaussian, probit, Poisson, NB2 and right-censored
piecewise-exponential survival models with independent Gaussian random intercepts.
The [structured interface](structured.md) adds alternative kernels, finite GP
bases, random slopes, Gaussian residual correlation and variance regression.
It also supports delayed entry, competing causes, known covariate paths and
left/interval censoring, with separate proper priors and a direct GP amplitude.
See [verified results and open gaps](reproduction.md) for the limits of the
numerical checks and original-study reproduction.
