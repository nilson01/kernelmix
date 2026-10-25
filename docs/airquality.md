# Ozone and weather in New York

This exploratory example models observed ozone using solar radiation, wind and
temperature. It includes source data, saved results and sampling checks. It is
separate from the applications in Bobb et al. (2015).

## Data and question

The [R airquality data](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html) contain 153 daily records from New York, May–September 1973. Ozone was measured at Roosevelt Island, solar radiation at Central Park, and wind and temperature at LaGuardia Airport. Their measurement times also differ. These are daily environmental measurements, not personal exposures.

| Variable | Role | Unit |
| --- | --- | --- |
| `Ozone` | Outcome, transformed with the natural logarithm | Parts per billion |
| `Solar.R` | Exposure function | Langleys |
| `Wind` | Exposure function | Miles per hour |
| `Temp` | Exposure function | Degrees Fahrenheit |
| `Month` | Linear adjustment, May as reference | Four indicators |

There are 37 missing ozone values and seven missing solar-radiation values. Joint complete-case filtering retains 111 days and excludes 42; no values are imputed. The [source CSV](data/airquality.csv) preserves all 153 records. Its [provenance](data/airquality-provenance.json) records the R version, hash, units and excluded row numbers.

| Month | Recorded days | Complete days |
| --- | --- | --- |
| May | 31 | 24 |
| June | 30 | 9 |
| July | 31 | 26 |
| August | 31 | 23 |
| September | 30 | 29 |

Ozone medians are 31 ppb in complete rows and 35 ppb in the five rows missing
solar radiation. Missingness is uneven across months, and ozone is unknown on
37 days. This comparison cannot establish ignorable missingness.

## Model and run

We use the Gaussian BKMR model with $Y_i=\log(\mathrm{Ozone}_i)$, $\mathbf z_i$ containing solar radiation, wind and temperature, and $\mathbf x_i$ containing an intercept and four month indicators:

$$Y_i=h(\mathbf z_i)+\mathbf x_i^\mathsf T\boldsymbol\beta+\epsilon_i,\qquad
\epsilon_i\overset{\mathrm{iid}}{\sim}N(0,\sigma^2).$$

Exposure standardization uses the 111 retained days. The Gaussian kernel uses independent $r_m\sim\mathrm{Gamma}(1,1)$ priors, with shape and rate equal to one. Residual variance has an inverse-gamma(2,1) prior. Other default Gaussian priors remain in force, including the flat adjustment-coefficient prior and $\lambda=\tau/\sigma^2\sim\mathrm{Gamma}(1,0.1)$. Exposure selection is disabled. Month adjustment handles coarse seasonality; independent errors are a working assumption, not a verified property of the daily series.

From the source folder, with KernelMix and its plotting extra installed:

```sh
python tools/analyze_airquality.py --draws 20000 --warmup 5000 --output ozone-results
python tools/plot_airquality.py --results ozone-results --output ozone-figure
```

Four chains run sequentially with seed 73. Each has 5,000 warmup iterations and 20,000 saved draws. An initial 4,000-draw run had maximum $\widehat R=1.0142$; one longer run was made to improve sampling, with the model, prior and data unchanged. This is an exploratory workflow, not a prospectively fixed study.

## Results and checks

| Check | Recorded result |
| --- | --- |
| Maximum rank/folded split $\widehat R$ | 1.0053 |
| Minimum bulk effective sample size | 885 |
| Minimum tail effective sample size | 1,193 |
| Consecutive-day residual pairs | 92 |
| Correlation of consecutive-day fitted residuals | 0.0136 |
| Fitting time on the recorded local environment | 72.6 seconds |

Diagnostics cover ten scalar parameters using all saved states; they do not
prove convergence of every curve. Curves and fitted residuals use 400 evenly
spaced states per chain. The residual correlation cannot establish independence.
Fitted-versus-observed values use training data, not held-out observations.

![Ozone example: (a–c) posterior means and 95% pointwise credible intervals for $h(\mathbf z)-h(\mathbf z_{0.5})$, with other exposures fixed at their sample medians and $\mathbf z_{0.5}$ the all-median profile; (d) fitted and observed log ozone; (e) residuals by date, with missing dates unconnected; (f) residual pairs one calendar day apart. Intervals assume independent Gaussian working errors.](figures/airquality.png)

The displayed slices increase with solar radiation and temperature and decrease with wind over much of the plotted range. These are conditional associations on the log scale. Curves use distinct marginal quantiles between the 5th and 95th percentiles; joint profiles can still be sparsely observed. The [curve table](results/airquality/curves.csv) includes distance to the nearest observed profile after standardization. Exponentiating a log-scale difference gives a ratio of fitted conditional medians under this model, not an arithmetic mean difference.

## Reuse and limits

The [results record](results/airquality/results.json), [parameter summaries](results/airquality/parameters.csv), [diagnostics](results/airquality/diagnostics.csv), [curve values](results/airquality/curves.csv) and [dated observations/residuals](results/airquality/observations.csv) accompany the runnable scripts. R is not required to run the analysis. Maintainers can regenerate the source CSV with `Rscript tools/export_airquality.R destination.csv`; the recorded export uses R 4.5.0.

Missingness, measurement differences, priors and serial dependence limit
interpretation. This example establishes neither causal effects nor time-series
interval coverage. Original-study reproduction and external research use remain
separate evidence gaps; see [reference studies](reproduction.md).
