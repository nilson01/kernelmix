"""Posterior arithmetic and serialization checks using fixed saved states."""
import json
import zipfile
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr

from kernelmix import BKMRFit, diagnose, overall_effect


def saved_fit(family="gaussian", *, zero=False, adjusted=True):
    z = np.array([[-1., 0.], [0., 1.], [1., -.5], [.5, .7]])
    raw = np.arange(4, dtype=float)[:, None] if adjusted else np.empty((4, 0))
    x = np.column_stack((np.ones(4), raw))
    shape = (2, 6)
    beta = np.zeros(shape + (x.shape[1],))
    beta[..., 0] = np.linspace(-.2, .3, 12).reshape(shape)
    if adjusted:
        beta[..., 1] = .1
    r = np.zeros(shape + (2,)) if zero else np.full(shape + (2,), .7)
    samples = {"beta": beta, "r": r, "lambda": np.ones(shape) * 1.4,
               "sigma2": np.ones(shape) * (1 if family == "binomial" else .4),
               "delta": r > 0, "pi": np.ones(shape)}
    y = np.array([0., 1., 0., 1.]) if family == "binomial" else np.array([.2, .8, -.4, 1.])
    if family == "binomial":
        samples["ystar"] = np.broadcast_to(np.array([-1., 2., -.5, .3]), shape + (4,)).copy()
    data = dict(y=y, Z=z, Z_original=z.copy(), X=x, X_original=raw,
                z_center=np.zeros(2), z_scale=np.ones(2), rows=np.arange(4),
                original_n=4, exposure_names=["lead", "zinc"],
                covariate_names=["Intercept", "age"] if adjusted else ["Intercept"], groups=None)
    settings = dict(family=family, chains=2, draws=6, intercept=True,
                    selection="none", r_fixed=[0, 0] if zero else None)
    return BKMRFit(data, samples, settings, {})


def test_summary_uses_saved_states_and_copies():
    fit = saved_fit()
    rows = {row["parameter"]: row for row in fit.summary_table(.8)}
    source = fit.samples["beta"][..., 0].ravel()
    row = rows["beta[Intercept]"]
    assert_allclose([row["mean"], row["sd"], row["lower"], row["upper"]],
                    [source.mean(), source.std(ddof=1), *np.quantile(source, [.1, .9])])
    copied = fit.posterior_samples("beta", combine_chains=True)
    copied[:] = -999
    assert not np.any(fit.samples["beta"] == -999)
    assert not fit.samples["beta"].flags.writeable
    assert "80% pointwise credible intervals" in fit.summary(.8)
    with pytest.raises(ValueError):
        fit.posterior_samples("not-a-parameter")
    with pytest.raises(ValueError):
        fit.inclusion_probabilities()


def test_identical_and_inactive_profiles_share_exact_h_draws():
    fit = saved_fit()
    prediction = fit.predict_h([[.1, .2], [.1, .2]], seed=42)
    assert_array_equal(prediction.draws[..., 0], prediction.draws[..., 1])
    zero = saved_fit(zero=True)
    prediction = zero.predict_h([[-10., 2.], [20., -7.]], seed=2)
    assert_array_equal(prediction.draws[..., 0], prediction.draws[..., 1])
    effect = overall_effect(zero, quantiles=[.1, .5, .9], seed=2)
    assert_array_equal(effect.draws, np.zeros_like(effect.draws))


def test_new_covariates_required_and_training_prediction_available():
    fit = saved_fit()
    with pytest.raises(ValueError, match="Xnew is required"):
        fit.predict([[.1, .2]])
    assert fit.predict(seed=2).draws.shape == (2, 6, 4)
    prediction = fit.predict([[.1, .2]], Xnew=[[2.]], seed=2)
    assert prediction.covariance.shape == (1, 1)
    assert len(prediction.to_records()) == 1
    with pytest.raises(ValueError):
        fit.predict([[.1, .2]], Xnew=[[2.], [3.]])


def test_probability_is_transform_of_each_joint_latent_draw():
    fit = saved_fit("binomial")
    options = dict(Znew=[[.1, .2], [.2, .4]], Xnew=[[1.], [2.]], seed=42)
    latent = fit.predict(scale="latent", **options)
    response = fit.predict(scale="response", **options)
    assert_allclose(response.draws, ndtr(latent.draws))
    observation = fit.predict(kind="observation", **options)
    assert np.isin(observation.draws, [0, 1]).all()
    # Transforming an average latent value would give a different result.
    assert np.any(np.abs(response.mean - ndtr(latent.mean)) > 1e-4)


def test_prediction_observation_adds_noise_and_seed_is_local():
    fit = saved_fit(adjusted=False)
    np.random.seed(31)
    before = np.random.get_state()
    mean = fit.predict([[0., .2]], seed=3)
    obs = fit.predict([[0., .2]], seed=3, kind="observation")
    after = np.random.get_state()
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert not np.array_equal(mean.draws, obs.draws)
    assert_array_equal(obs.draws, fit.predict([[0., .2]], seed=3, kind="observation").draws)


@pytest.mark.parametrize("probability", [0, 1, -.1, np.nan, np.inf, True, [.95]])
def test_probability_errors(probability):
    fit = saved_fit()
    with pytest.raises(ValueError):
        fit.summary_table(probability)
    with pytest.raises(ValueError):
        fit.predict_h([[0., .2]], probability=probability)


def test_group_pips_are_distinct_and_undefined_when_never_included():
    fit = saved_fit()
    data, samples, settings = deepcopy(fit.data), fit.posterior_samples(), deepcopy(fit.settings)
    data["groups"] = ["metals", "metals"]
    samples["delta"][:] = False
    samples["delta"][0, :3, 0] = True
    samples["delta"][1, :3, 1] = True
    settings["selection"] = "group"
    grouped = BKMRFit(data, samples, settings, {})
    rows = grouped.inclusion_probabilities()
    assert_allclose([rows[0]["pip"], rows[0]["group_pip"], rows[0]["conditional_pip"]], [.25, .5, .5])
    samples["delta"][:] = False
    never = BKMRFit(data, samples, settings, {}).inclusion_probabilities()
    assert np.isnan(never[0]["conditional_pip"])


def test_safe_archive_round_trip_and_invalid_inputs(tmp_path):
    fit = saved_fit("binomial")
    path = tmp_path / "fit.kmix"
    fit.save(path)
    restored = BKMRFit.load(path)
    assert_allclose(restored.predict_h([[0., .2]], seed=3).draws,
                    fit.predict_h([[0., .2]], seed=3).draws)
    assert restored.settings == fit.settings
    with pytest.raises(ValueError, match="exceeds"):
        BKMRFit.load(path, max_bytes=1)
    with zipfile.ZipFile(path) as archive:
        assert all(name.endswith(".npy") for name in archive.namelist())
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, metadata=np.asarray(json.dumps({"schema": 999})))
    with pytest.raises(ValueError, match="schema"):
        BKMRFit.load(invalid)
    np.savez(invalid, metadata=np.asarray(json.dumps({"schema": 1, "data": {}})),
             unknown=np.array([1]))
    with pytest.raises(ValueError, match="unknown"):
        BKMRFit.load(invalid)


def test_diagnostics_detect_shift_and_correlation():
    rng = np.random.default_rng(91)
    independent = rng.normal(size=(4, 1000))
    iid = diagnose(independent)
    shifted = independent + np.arange(4)[:, None] * 3
    bad = diagnose(shifted)
    assert iid["rhat"] < 1.02
    assert bad["rhat"] > 1.2
    correlated = np.zeros((4, 1000))
    for index in range(1, 1000):
        correlated[:, index] = .95 * correlated[:, index - 1] + independent[:, index]
    assert diagnose(correlated)["ess_bulk"] < iid["ess_bulk"] / 4
    assert np.isnan(diagnose(np.ones((2, 20)))["rhat"])
    assert np.isnan(diagnose(independent[:1])["rhat"])
    with pytest.raises(ValueError):
        diagnose([1, 2])


def test_constant_sampled_parameter_is_reported_but_fixed_variance_is_not():
    result = saved_fit('binomial')
    report = result.diagnostics()
    assert any('lambda: constant' in message for message in report['messages'])
    assert 'sigma2' not in [row['parameter'] for row in report['parameters']]


def test_probit_saved_state_signs_and_variance_are_validated():
    result = saved_fit('binomial')
    samples = result.posterior_samples()
    samples['sigma2'][:] = 2
    with pytest.raises(ValueError, match='variance'):
        BKMRFit(result.data, samples, result.settings, {})
    samples = result.posterior_samples()
    samples['ystar'][0, 0, 0] = 1
    with pytest.raises(ValueError, match='signs'):
        BKMRFit(result.data, samples, result.settings, {})


def test_prediction_memory_is_checked_before_dense_allocation():
    result = saved_fit(adjusted=False)
    large = np.zeros((4000, 2))
    with pytest.raises(ValueError, match='working-memory'):
        result.predict_h(large, seed=1)
    with pytest.raises(ValueError, match='working-memory'):
        result.posterior_moments_h(large)


def test_selection_summary_handles_boolean_indicators_and_diagnoses_inclusion():
    result = saved_fit()
    settings = deepcopy(result.settings)
    settings['selection'] = 'component'
    selected = BKMRFit(result.data, result.posterior_samples(), settings, {})
    rows = {row['parameter']: row for row in selected.summary_table()}
    assert rows['delta[lead]']['mean'] == 1
    assert rows['delta[lead]']['lower'] == 1
    assert 'Posterior inclusion probabilities' in selected.summary()
    parameters = [row['parameter'] for row in selected.diagnostics()['parameters']]
    assert 'pi' in parameters and 'delta[lead]' in parameters
