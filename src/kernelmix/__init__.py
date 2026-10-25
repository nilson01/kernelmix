"""Bayesian kernel machine regression for exposure mixtures."""
from ._kernel import gaussian_kernel
from .diagnostics import diagnose
from .effects import (
    CurveResult,
    EffectResult,
    SurfaceResult,
    bivariate_response,
    interaction_effects,
    overall_effect,
    single_exposure_effects,
    univariate_response,
)
from .model import fit
from .plots import plot_overall, plot_pips, plot_response, plot_surface, plot_trace
from .priors import Prior
from .result import BKMRFit, Prediction

__version__ = "0.1.0"
__all__ = [
    "fit", "Prior", "BKMRFit", "Prediction", "gaussian_kernel", "diagnose",
    "EffectResult", "CurveResult", "SurfaceResult", "overall_effect",
    "univariate_response", "bivariate_response", "single_exposure_effects",
    "interaction_effects", "plot_trace", "plot_pips", "plot_response",
    "plot_overall", "plot_surface",
]
