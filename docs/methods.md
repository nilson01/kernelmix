# Model and interpretation

## Continuous outcomes

We first describe a continuous outcome. Let $Y_i$ be the outcome for observation $i$, $\mathbf z_i$ its $M$ exposures, and $\mathbf x_i$ its adjustment covariates. Following [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058), the function $h(\mathbf z_i)$ describes the joint exposure contribution, while $\mathbf x_i^{\mathsf T}\boldsymbol\beta$ provides linear adjustment:

$$
Y_i=h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta+\epsilon_i,
\qquad \epsilon_i\overset{\mathrm{iid}}{\sim}N(0,\sigma^2).
$$

The error $\epsilon_i$ is independent normal noise with variance $\sigma^2$. The function $h$ allows curved associations and interactions, where the association with one exposure depends on another.

To relate values of $h$ at different exposure profiles (rows of exposure values), we use a Gaussian process (GP) prior. A prior specifies their distribution before using outcome data. At the observed profiles,

$$
\mathbf h=(h_1,\ldots,h_n)^{\mathsf T}\sim N(\mathbf0,\tau\mathbf K_{Z,\mathbf r}),
\qquad
K(\mathbf z,\mathbf z';\mathbf r)=
\exp\left\{-\sum_{m=1}^{M}r_m(z_m-z'_m)^2\right\}.
$$

Here $h_i=h(\mathbf z_i)$, $n$ is the observation count, $\tau$ controls prior variation in $h$, and $\mathbf K_{Z,\mathbf r}$ contains similarities between observed profiles. Each relevance parameter $r_m\geq0$ controls how exposure $m$ changes similarity; $r_m=0$ removes it from $h$.

The default Gaussian sampler integrates out $\mathbf h$: it accounts for its distribution without sampling each value during parameter fitting. It stores $\lambda=\tau/\sigma^2$ as `lambda`, giving covariance $\sigma^2(\mathbf I+\lambda\mathbf K_{Z,\mathbf r})$, where $\mathbf I$ is the identity matrix.

The code uses the following names for the paper's quantities. The meaning of `lambda` depends on the fitting route.

| Paper quantity | Meaning | API name |
| --- | --- | --- |
| $\mathbf z_i$, $\mathbf x_i$ | Exposure and adjustment rows | Rows of `Z` and `X` |
| $h$, $\boldsymbol\beta$ | Exposure function and adjustment coefficients | `predict_h`, `beta` |
| $\sigma^2$, $r_m$ | Residual variance and exposure relevance | `sigma2`, `r` |
| $\tau$ | GP covariance amplitude | Default Gaussian/probit: `lambda` stores $\tau/\sigma^2$; structured and default count/survival: it stores $\tau$ directly |

The [structured guide](structured.md) defines its separate priors. Equal numerical values of `lambda` need not describe the same covariance across these routes.

## Binary outcomes

For a binary outcome, the exposure function and linear adjustment instead determine a probability. With `family="binomial"`, the probit link gives

$$
\Pr(Y_i=1\mid h,\boldsymbol\beta)=
\Phi\{h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta\}.
$$

Here $\Phi$ is the standard normal cumulative distribution function. The binary outcome is determined by whether an underlying normal variable exceeds zero; its residual variance is fixed at one. This extension follows [Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y). Probability contrasts depend on exposures and adjustment covariates and are not odds ratios.

The coefficient prior also matters. Default fits use a flat prior and reject complete or quasi-complete separation, where adjustment covariates separate outcome classes, possibly with observations on the separating boundary. The posterior may then have infinite total probability. Structured fits instead use a proper normal coefficient prior, with total probability one.

## Count outcomes

For counts, the same exposure and adjustment contributions enter the log mean. A random intercept $u_{c(i)}$ can also account for observation $i$'s cluster $c(i)$; it is zero without clustering. For Poisson and negative-binomial (NB2) models, define

$$
\eta_i=h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta+u_{c(i)},
\qquad \mu_i=\exp(\eta_i+o_i).
$$

The known log offset $o_i$ accounts for observation time or exposure amount. Conditional on the predictors and cluster effect, the Poisson mean and variance are both $\mu_i$. NB2 allows extra variation:

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

For event times, the predictor $\eta_i$ determines a relative hazard rather than a count mean. Using $\eta_i$ above without a count offset, the model is

$$
\lambda_i(t)=\lambda_0(t)\exp(\eta_i),\qquad
S_i(t)=\exp\{-\Lambda_0(t)\exp(\eta_i)\}.
$$

A hazard is the instantaneous event rate among subjects still event-free. Here $\lambda_0(t)$ is the baseline hazard, $\Lambda_0(t)$ its integral, and $S_i(t)$ the probability of remaining event-free through $t$.

Each record supplies a positive observed time $t_i$ and an indicator $d_i$: one for an observed event, zero when follow-up ends first (right censoring). Under independent right censoring, its likelihood contribution is $\{\lambda_i(t_i)\}^{d_i}S_i(t_i)$. We use constant baseline rates $\lambda_{0j}$ within right-closed intervals. Let $D_{ij}$ be subject $i$'s time in interval $j$, $j(i)$ its event interval, and $d_j$ the event count in interval $j$. Then

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

The outcome models above can share variation within observation clusters. In the default random-intercept route, $c(i)$ identifies observation $i$'s cluster and $u_g\sim N(0,\sigma_u^2)$ independently across clusters. Conditional on $h$, covariates and these intercepts, observations are independent under the outcome likelihood. The `cluster` labels identify observations; `groups` instead groups exposures for selection.

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

For prediction, known labels use fitted intercepts, while equal unseen labels share one new Gaussian draw. Omitting labels fixes $u=0$, rather than averaging over cluster effects. Random intercepts do not provide arbitrary serial correlation or cluster-robust inference. Constant kernels or clusters with one observation may leave little information to distinguish GP variation from cluster variation.

## Priors and exposure selection

The prior completes the model specification. Default Gaussian/probit priors follow R `bkmr` 0.2.2; [structured priors](structured.md) differ. Gamma distributions below use shape and **rate**. The inverse-gamma density is proportional to $(\sigma^2)^{-a-1}\exp(-b/\sigma^2)$ for shape $a$ and parameter $b$.

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

The relevance prior needs particular care: inverse-uniform $r_m$ has an unbounded upper tail, and a finite sampling average does not establish that its posterior mean exists. Inspect quantiles and mixing and assess prior sensitivity. `Prior(r_distribution="gamma")` and `"unif"` provide alternatives. Gamma parameters use mean and standard deviation; uniform bounds apply to $r_m$ for `"unif"` and $1/r_m$ for `"invunif"`.

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

After preprocessing, each chain samples the posterior distribution, which combines data and priors. Chains run sequentially with separate random streams. `draws` counts saved draws after `warmup`, and `thin` sets their spacing. Proposals remain fixed; `sampling_info` records acceptance counts and runtime.

The sampler depends on the outcome. Default Gaussian fits integrate out $h$ during parameter sampling. Binary fits also sample underlying normal outcomes constrained by the observed response signs; prediction uses these saved values.

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

The fitted model can return either the exposure contribution or the full outcome prediction. `predict_h` draws $h$ jointly across profiles. `predict(kind="mean")` returns Gaussian/count means, binary probabilities or survival relative hazards. `predict_survival` returns survival probabilities or hazards on a time grid. `predict(kind="observation")` adds new outcome variation for Gaussian, binary and response-scale count predictions; survival observation sampling is unavailable. The binary latent scale is the underlying normal scale.

To compare exposure profiles, contrasts subtract **joint** posterior draws, retaining covariance between predictions. `overall_effect` compares all-exposure quantile profiles with a reference; `single_exposure_effects` changes one exposure with others fixed; `interaction_effects` compares two single-exposure contrasts. Curves and surfaces vary one or two exposures with others fixed.

Equal-tailed intervals leave the same posterior probability below and above their limits. They apply to individual points, not the whole curve simultaneously. Response-scale summaries fix adjustment covariates at retained training means unless `X` supplies another profile.

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
