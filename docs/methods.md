# Model and interpretation

## Continuous outcomes

Following [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058),

$$
Y_i=h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta+\epsilon_i,
\qquad \epsilon_i\overset{\mathrm{iid}}{\sim}N(0,\sigma^2).
$$

Here, $Y_i$ is the outcome, $\mathbf z_i$ contains the $M$ exposures,
$\mathbf x_i$ contains adjustment covariates, and $\boldsymbol\beta$ contains
their coefficients. The function $h$ allows nonlinear exposure associations
and interactions. Its prior at the observed exposure profiles is

$$
\mathbf h=(h_1,\ldots,h_n)^{\mathsf T}\sim N(\mathbf0,\tau\mathbf K_{Z,\mathbf r}),
\qquad
K(\mathbf z,\mathbf z';\mathbf r)=
\exp\left\{-\sum_{m=1}^{M}r_m(z_m-z'_m)^2\right\}.
$$

The relevance parameters $r_m\geq0$ determine how each exposure changes the
kernel. An exposure with $r_m=0$ does not affect $h$. In the software,
`lambda` is $\lambda=\tau/\sigma^2$. Integrating out $\mathbf h$ gives
covariance $\sigma^2(\mathbf I+\lambda\mathbf K_{Z,\mathbf r})$.

## Binary outcomes

With `family="binomial"`, the model uses a probit link:

$$
\Pr(Y_i=1\mid h,\boldsymbol\beta)=
\Phi\{h(\mathbf z_i)+\mathbf x_i^{\mathsf T}\boldsymbol\beta\}.
$$

Here $\Phi$ is the standard normal cumulative distribution function. Residual
variance on the latent normal scale is fixed at one. This extension follows
[Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y).
Probability contrasts depend on both exposure profiles and adjustment covariates.
They are not odds ratios. Separated linear adjustment designs are rejected:
the flat coefficient prior can give an improper posterior in this case.

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
They extend KernelMix beyond R `bkmr` 0.2.2; they are not described as R BKMR
implementations or new statistical methods.

## Survival outcomes

For positive observed time $t_i$ and event indicator $d_i$, the model is

$$
\lambda_i(t)=\lambda_0(t)\exp(\eta_i),\qquad
S_i(t)=\exp\{-\Lambda_0(t)\exp(\eta_i)\}.
$$

Here $\lambda_0(t)$ is the baseline hazard and $\Lambda_0(t)$ its integral.
The linear predictor $\eta_i$ is defined above without a count offset.
Independent right censoring gives likelihood contribution
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
prior. Cutpoints are conditioned on after selection; their uncertainty is not
modeled. Defaults use event-time deciles, so assess sensitivity to their number
and placement. Baseline priors depend on the chosen time unit.

No delayed entry, competing risks, time-varying covariates, left censoring or
interval censoring is implemented. All-censored data are accepted with a warning:
the posterior then depends strongly on the priors. A fitted intercept and the
baseline hazard share a scale. Separate estimates are anchored by their priors;
`intercept=False` avoids adding that redundant coefficient.

## Random intercepts

With observation-cluster index $c(i)$, independent Gaussian intercepts satisfy
$u_g\sim N(0,\sigma_u^2)$. Exposures in `groups` and observations in `cluster`
serve different purposes. Conditional on $h$, covariates and random intercepts,
observations are independent under the selected outcome likelihood.

For Gaussian/probit fits, `lambda_cluster` is $\sigma_u^2/\sigma^2$, with probit
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
Its use with the new likelihoods is an additional model composition.

Known-cluster prediction conditions on the cluster posterior. Equal unseen labels
share a new Gaussian draw. Omitted labels fix $u=0$; this is not marginalization
over the cluster distribution. Random intercepts do not provide arbitrary serial
correlation or cluster-robust inference. GP and cluster variance components can
be weakly separated, particularly for constant kernels or singleton clusters.

## Priors and exposure selection

`Prior()` and Gaussian/probit fitting defaults follow R `bkmr` 0.2.2. Gamma distributions below use shape and
**rate**. The inverse-gamma density is proportional to
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
At fixed finite predictor values, these defaults give finite first and second
prior-predictive moments of the count mean and relative hazard. A supplied `Prior` is used as specified. A partial dictionary overlays
the selected family's defaults, for example `prior={"r_distribution": "gamma"}`.
Warnings identify supplied amplitude or dispersion priors for which these moment
guarantees do not hold. Assess prior sensitivity; large finite moments can still
be numerically unstable. `sigma2` is only a stored unit-variance convention for
new families and is omitted from their summaries.

The inverse-uniform relevance prior has an unbounded upper tail. A finite-chain
average of $r_m$ does not imply that its posterior expectation exists. Inspect
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
across the outcome, exposures, covariates, cluster labels, and applicable offsets or event indicators. Retained original row indices are
stored in `result.data["rows"]`. Complete-case fitting does not correct bias
caused by missing data. Covariates must be numeric; categorical encoding is the
caller's responsibility.

Chains run sequentially with separate random streams. `draws` counts saved draws
per chain after `warmup`; `thin` controls the spacing of saved states. Proposals
are fixed during sampling. `sampling_info` records acceptance counts and runtime.
The Gaussian model integrates out $h$ during parameter sampling. The binary
model also samples correlated, sign-constrained latent outcomes. Predictions
condition on the saved latent outcomes for binary fits.

Count/survival fits sample a noncentered GP with
[elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html),
log-scale Metropolis updates and conjugate baseline updates. Eigenvalues below
$100n\epsilon\max(1,\lambda_{\max}(\mathbf K))$, where $\epsilon$ is machine
precision, define numerical null directions. No diagonal jitter is added.
Prediction uses the same rank rule and the saved latent function. Saved auxiliary
white coordinates may use different eigenvector signs on another platform;
archive validation checks mathematical support rather than those signs.

Dense matrix factorization requires roughly cubic work in $n$ per parameter
update. Working-memory checks reject oversized fitting or prediction requests;
they are protective estimates, not a promise about peak process memory.

## Prediction and contrasts

`predict_h` draws $h$ jointly over requested profiles. `predict(kind="mean")`
returns outcome means, binary probabilities, or relative hazards for survival.
`predict_survival` gives survival probabilities or hazards on a time grid.
`predict(kind="observation")` also includes new observation variation for
Gaussian, binary and response-scale count predictions. Survival observation
sampling is not implemented.
The latent binary scale is the underlying normal scale.

Contrasts subtract **joint** posterior draws so that covariance is retained.
`overall_effect` compares all-exposure quantile profiles with a reference
profile. `single_exposure_effects` changes one exposure while holding the others
fixed. `interaction_effects` returns a difference of two single-exposure
contrasts. Curves and surfaces vary one or two exposures; others are held fixed.
Their intervals are equal-tailed, pointwise credible intervals, not simultaneous
bands. On the response scale, adjustment covariates default to their retained
training means unless `X` supplies a different profile.

Marginal quantile profiles may be uncommon in correlated data. Check exposure
support before interpreting these contrasts. Causal interpretation requires
additional identification assumptions; fitting a model does not supply them.

## Diagnostics and scope

`diagnostics()` reports rank-normalized/folded split $\widehat R$, bulk effective
sample size and tail effective sample size. Large $\widehat R$, small effective
sample sizes, or constant sampled traces need investigation. Fixed parameters
are excluded from these diagnostics. Trace plots and sensitivity checks remain
necessary; favorable diagnostics do not prove convergence.

KernelMix supports Gaussian, probit, Poisson, NB2 and right-censored
piecewise-exponential survival models with independent Gaussian random intercepts.
Low-rank speed approximations, random slopes and more general correlation
structures remain unsupported. Numerical comparisons cover selected
fixed states and small analytic posterior targets. They do not establish general
frequentist coverage or reproduce the original paper's studies.
