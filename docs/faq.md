# Common questions

## Do I need R?

No. NumPy and SciPy handle fitting and prediction. R is needed only to regenerate the reference comparisons or export the original R example dataset; see [development](development.md).

## Which outcome model should I use?

Use Gaussian for continuous outcomes, probit (`family="binomial"`) for zero/one outcomes, Poisson or negative binomial for counts, and survival for event times. The [model guide](methods.md) states the likelihoods and assumptions; [fitting inputs](api.md#fitting) lists the required arguments.

## Why does prediction require `Xnew`?

If a fit includes adjustment covariates, prediction at new exposure rows needs corresponding adjustment values. Supply the raw columns in training order, without the added intercept. `predict_h` returns only the exposure function and does not require those adjustments. [Prediction](api.md#prediction) explains the distinction.

## How are missing values handled?

The default raises an error. `missing="drop"` removes rows with missing required values; for recorded subject histories, it removes the whole affected history. Dropping incomplete observations does not correct missing-data bias. See [input handling](methods.md#inputs-and-computation).

## What does `lambda` mean?

Default Gaussian fits store the variance ratio $\lambda=\tau/\sigma^2$, where $\tau$ controls Gaussian process (GP) variation and $\sigma^2$ is residual variance. Default probit fits fix $\sigma^2=1$. Structured and default count/survival fits store the direct GP amplitude $\tau$. Compare priors on the stated scales; see the [notation/API mapping](methods.md#continuous-outcomes).

## Are exposure curves causal effects?

They describe fitted associations at specified values of other variables. A causal interpretation needs additional scientific assumptions. A percentile combination can also be rare in observed data, especially with correlated exposures. See [contrasts and curves](api.md#exposure-contrasts-and-curves).

## What is the difference between a mean and an observation prediction?

A mean prediction describes the fitted mean, probability or relative hazard, according to the outcome family. An observation prediction also includes new outcome variation; it is unavailable for survival outcomes. [Prediction scales](api.md#prediction) gives the supported combinations.

## Is a survival contrast a hazard ratio?

Response-scale survival contrasts subtract relative hazards. To obtain a hazard ratio, exponentiate each latent exposure-function contrast before summarizing the draws. Survival probabilities come from `predict_survival`; see the [survival tutorial](tutorial.md#fit-right-censored-survival-times).

## Why is a group's conditional PIP `NaN`?

A conditional posterior inclusion probability asks how often an exposure is included when its group is selected. If the group is never selected, that fraction is undefined. Group selection permits at most one active exposure in a group. See [selection priors](methods.md#priors-and-exposure-selection).

## Do favorable diagnostics validate the model?

No. R-hat and effective sample sizes assess sampling behavior. They cannot establish convergence of every curve, independence of daily errors, or correctness of model assumptions. See [diagnostics](methods.md#diagnostics-and-scope) and [evidence](reproduction.md).

## Do finite GP bases preserve uncertainty from the full model?

Not in general. The predictive process omits variation remaining after its knot values are known. Random Fourier features condition inference on a fixed sampled feature map. These approximations can change uncertainty; see [GP representations](structured.md#reduce-gp-matrix-size).

## Does KernelMix reproduce the original BKMR studies?

Exact reproduction remains unresolved because original study inputs and sufficient fitting settings are missing. The ozone example and numerical software checks answer different questions. The [evidence page](reproduction.md) records their scope and limits.
