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

## Priors and exposure selection

`Prior()` defaults match R `bkmr` 0.2.2. Gamma distributions below use shape and
**rate**. The inverse-gamma density is proportional to
$(\sigma^2)^{-a-1}\exp(-b/\sigma^2)$ for shape $a$ and parameter $b$.

| Quantity | Default |
| --- | --- |
| $\lambda$ | Gamma(1, 0.1) |
| $\sigma^2$, continuous outcome | Inverse-gamma(0.001, 0.001) |
| Active $r_m$ | $1/r_m\sim\mathrm{Uniform}(0,100)$ |
| Inclusion probability $\pi$ | Beta(1,1) |
| $\boldsymbol\beta$ | Flat prior |

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
across the outcome, exposures and covariates. Retained original row indices are
stored in `result.data["rows"]`. Complete-case fitting does not correct bias
caused by missing data. Covariates must be numeric; categorical encoding is the
caller's responsibility.

Chains run sequentially with separate random streams. `draws` counts saved draws
per chain after `warmup`; `thin` controls the spacing of saved states. Proposals
are fixed during sampling. `sampling_info` records acceptance counts and runtime.
The Gaussian model integrates out $h$ during parameter sampling. The binary
model also samples correlated, sign-constrained latent outcomes. Predictions
condition on the saved latent outcomes for binary fits.

Dense matrix factorization requires roughly cubic work in $n$ per parameter
update. Working-memory checks reject oversized fitting or prediction requests;
they are protective estimates, not a promise about peak process memory.

## Prediction and contrasts

`predict_h` draws $h$ jointly over requested profiles. `predict(kind="mean")`
returns outcome means, or probabilities for binary outcomes.
`predict(kind="observation")` also includes new observation variation.
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

KernelMix supports the dense Gaussian kernel with continuous and binary outcomes.
It does not implement random intercepts, clustered outcomes, low-rank
approximations, count or survival outcomes. Numerical comparisons cover selected
fixed states and small analytic posterior targets. They do not establish general
frequentist coverage or reproduce the original paper's studies.
