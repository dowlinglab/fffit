import pytest
import numpy


from fffit.utils import values_real_to_scaled
from fffit.tests.base_test import BaseTest

class TestUtils(BaseTest):
    def test_values_real_to_scaled(self):
        bounds = [2., 4.]
        value = 2.0
        scaled_value = values_real_to_scaled(value, bounds)
        assert np.isclose(scaled_value, 0.0)

        value = 4.0
        scaled_value = values_real_to_scaled(value, bounds)
        assert np.isclose(scaled_value, 1.0)

