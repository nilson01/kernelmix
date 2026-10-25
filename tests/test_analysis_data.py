"""Independent data-contract checks for the documented observed-data example."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def analysis_module():
    script = Path(__file__).parents[1] / 'tools/analyze_airquality.py'
    spec = importlib.util.spec_from_file_location('airquality_analysis', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missingness_and_month_design_match_r_data(tmp_path):
    module = analysis_module()
    source = Path(__file__).parents[1] / 'docs/data/airquality.csv'
    if not source.exists():
        # Installed-wheel CI copies tools/tests, not user documentation. A fixture
        # mirrors the immutable published dataset for that environment.
        source = Path(__file__).parent / 'fixtures/airquality.csv'
    retained, design, keep, original, profile = module.load_data(source)
    assert len(original) == 153 and len(retained) == 111
    assert profile['missing_by_field']['Ozone'] == 37
    assert profile['missing_by_field']['Solar.R'] == 7
    assert [r['complete'] for r in profile['month_counts']] == [24, 9, 26, 23, 29]
    np.testing.assert_array_equal(design, retained[:, 4, None] == np.arange(6, 10))
    np.testing.assert_array_equal(original[keep], retained)
    assert np.linalg.matrix_rank(np.column_stack((np.ones(111), design))) == 5


def test_changed_data_are_rejected_before_fitting(tmp_path):
    module = analysis_module()
    changed = tmp_path / 'changed.csv'
    changed.write_text('Ozone,Solar.R,Wind,Temp,Month,Day\n0,1,2,3,5,1\n')
    with pytest.raises(ValueError, match='differs from the documented'):
        module.load_data(changed)
