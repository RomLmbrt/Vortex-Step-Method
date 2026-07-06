import os
import sys

import numpy as np

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, "src"))

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.WingGeometry import Wing


def _inviscid_polar_data():
    alpha = np.deg2rad(np.arange(-10, 31, 1))
    return np.column_stack(
        (
            alpha,
            2 * np.pi * alpha,
            np.zeros_like(alpha),
            np.zeros_like(alpha),
        )
    )


def _rectangular_body(y_offset=0.0):
    wing = Wing(n_panels=4, spanwise_panel_distribution="unchanged")
    polar_data = _inviscid_polar_data()

    for y in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        wing.add_section(
            np.array([0.0, y + y_offset, 0.0]),
            np.array([1.0, y + y_offset, 0.0]),
            polar_data,
        )

    return BodyAerodynamics([wing])


def test_elliptical_gamma_initialization_is_invariant_to_y_translation():
    centered_body = _rectangular_body(y_offset=0.0)
    translated_body = _rectangular_body(y_offset=10.0)

    centered_gamma = centered_body.compute_circulation_distribution_elliptical_wing()
    translated_gamma = translated_body.compute_circulation_distribution_elliptical_wing()

    assert np.all(np.isfinite(translated_gamma))
    np.testing.assert_allclose(translated_gamma, centered_gamma)
