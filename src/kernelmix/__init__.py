"""Bayesian kernel machine regression for exposure mixtures."""
from ._basis import kernel_matrix
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
from .plots import plot_overall, plot_pips, plot_response, plot_surface, plot_survival, plot_trace
from .priors import Prior
from .result import BKMRFit, Prediction, SurvivalPrediction
from .structured_result import CompetingRiskPrediction, RandomCoefficientPrediction, StructuredFit

__version__ = "0.3.0"
__all__ = [
    "StructuredFit", "RandomCoefficientPrediction", "CompetingRiskPrediction", "kernel_matrix",
    "fit", "Prior", "BKMRFit", "Prediction", "gaussian_kernel", "diagnose",
    "EffectResult", "CurveResult", "SurfaceResult", "overall_effect",
    "univariate_response", "bivariate_response", "single_exposure_effects",
    "interaction_effects", "plot_trace", "plot_pips", "plot_response",
    "plot_overall", "plot_surface", "plot_survival", "SurvivalPrediction",
]
