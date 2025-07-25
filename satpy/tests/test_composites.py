#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018-2020 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for compositors in composites/__init__.py."""

import datetime as dt
import os
import unittest
from unittest import mock

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition

import satpy
from satpy.tests.utils import RANDOM_GEN, CustomScheduler

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


class TestMatchDataArrays:
    """Test the utility method 'match_data_arrays'."""

    def _get_test_ds(self, shape=(50, 100), dims=("y", "x")):
        """Get a fake DataArray."""
        from pyresample.geometry import AreaDefinition
        data = da.random.random(shape, chunks=25)
        area = AreaDefinition(
            "test", "test", "test",
            {"proj": "eqc", "lon_0": 0.0,
             "lat_0": 0.0},
            shape[dims.index("x")], shape[dims.index("y")],
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        attrs = {"area": area}
        return xr.DataArray(data, dims=dims, attrs=attrs)

    def test_single_ds(self):
        """Test a single dataset is returned unharmed."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays((ds1,))
        assert ret_datasets[0].identical(ds1)

    def test_mult_ds_area(self):
        """Test multiple datasets successfully pass."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays((ds1, ds2))
        assert ret_datasets[0].identical(ds1)
        assert ret_datasets[1].identical(ds2)

    def test_mult_ds_no_area(self):
        """Test that all datasets must have an area attribute."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        del ds2.attrs["area"]
        comp = CompositeBase("test_comp")
        with pytest.raises(ValueError, match="Missing 'area' attribute"):
            comp.match_data_arrays((ds1, ds2))

    def test_mult_ds_diff_area(self):
        """Test that datasets with different areas fail."""
        from pyresample.geometry import AreaDefinition

        from satpy.composites import CompositeBase, IncompatibleAreas
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        ds2.attrs["area"] = AreaDefinition(
            "test", "test", "test",
            {"proj": "eqc", "lon_0": 0.0,
             "lat_0": 0.0},
            100, 50,
            (-30037508.34, -20018754.17, 10037508.34, 18754.17))
        comp = CompositeBase("test_comp")
        with pytest.raises(IncompatibleAreas):
            comp.match_data_arrays((ds1, ds2))

    def test_mult_ds_diff_dims(self):
        """Test that datasets with different dimensions still pass."""
        from satpy.composites import CompositeBase

        # x is still 50, y is still 100, even though they are in
        # different order
        ds1 = self._get_test_ds(shape=(50, 100), dims=("y", "x"))
        ds2 = self._get_test_ds(shape=(3, 100, 50), dims=("bands", "x", "y"))
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays((ds1, ds2))
        assert ret_datasets[0].identical(ds1)
        assert ret_datasets[1].identical(ds2)

    def test_mult_ds_diff_size(self):
        """Test that datasets with different sizes fail."""
        from satpy.composites import CompositeBase, IncompatibleAreas

        # x is 50 in this one, 100 in ds2
        # y is 100 in this one, 50 in ds2
        ds1 = self._get_test_ds(shape=(50, 100), dims=("x", "y"))
        ds2 = self._get_test_ds(shape=(3, 50, 100), dims=("bands", "y", "x"))
        comp = CompositeBase("test_comp")
        with pytest.raises(IncompatibleAreas):
            comp.match_data_arrays((ds1, ds2))

    def test_nondimensional_coords(self):
        """Test the removal of non-dimensional coordinates when compositing."""
        from satpy.composites import CompositeBase
        ds = self._get_test_ds(shape=(2, 2))
        ds["acq_time"] = ("y", [0, 1])
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays([ds, ds])
        assert "acq_time" not in ret_datasets[0].coords

    def test_almost_equal_geo_coordinates(self):
        """Test that coordinates that are almost-equal still match.

        See https://github.com/pytroll/satpy/issues/2668 for discussion.

        Various operations like cropping and resampling can cause
        geo-coordinates (y, x) to be very slightly unequal due to floating
        point precision. This test makes sure that even in those cases we
        can still generate composites from DataArrays with these coordinates.

        """
        from satpy.composites import CompositeBase
        from satpy.coords import add_crs_xy_coords

        comp = CompositeBase("test_comp")
        data_arr1 = self._get_test_ds(shape=(2, 2))
        data_arr1 = add_crs_xy_coords(data_arr1, data_arr1.attrs["area"])
        data_arr2 = self._get_test_ds(shape=(2, 2))
        data_arr2 = data_arr2.assign_coords(
            x=data_arr1.coords["x"] + 0.000001,
            y=data_arr1.coords["y"],
            crs=data_arr1.coords["crs"],
        )
        # data_arr2 = add_crs_xy_coords(data_arr2, data_arr2.attrs["area"])
        # data_arr2.assign_coords(x=data_arr2.coords["x"].copy() + 1.1)
        # default xarray alignment would fail and collapse one of our dims
        assert 0 in (data_arr2 - data_arr1).shape
        new_data_arr1, new_data_arr2 = comp.match_data_arrays([data_arr1, data_arr2])
        assert 0 not in new_data_arr1.shape
        assert 0 not in new_data_arr2.shape
        assert 0 not in (new_data_arr2 - new_data_arr1).shape


class TestRatioSharpenedCompositors:
    """Test RatioSharpenedRGB and SelfSharpendRGB compositors."""

    def setup_method(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition("test", "test", "test",
                              {"proj": "merc"}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        attrs = {"area": area,
                 "start_time": dt.datetime(2018, 1, 1, 18),
                 "modifiers": tuple(),
                 "resolution": 1000,
                 "calibration": "reflectance",
                 "units": "%",
                 "name": "test_vis"}
        low_res_data = np.ones((2, 2), dtype=np.float64) + 4
        low_res_data[1, 1] = 0.0  # produces infinite ratio
        ds1 = xr.DataArray(da.from_array(low_res_data, chunks=2),
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        self.ds1 = ds1

        ds2 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 2,
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds2.attrs["name"] += "2"
        self.ds2 = ds2

        ds3 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 3,
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds3.attrs["name"] += "3"
        self.ds3 = ds3

        # high resolution version
        high_res_data = np.ones((2, 2), dtype=np.float64)
        high_res_data[1, 0] = np.nan  # invalid value in one band
        ds4 = xr.DataArray(da.from_array(high_res_data, chunks=2),
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds4.attrs["name"] += "4"
        ds4.attrs["resolution"] = 500
        self.ds4 = ds4

        # high resolution version - but too big
        ds4_big = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64),
                               attrs=attrs.copy(), dims=("y", "x"),
                               coords={"y": [0, 1, 2, 3], "x": [0, 1, 2, 3]})
        ds4_big.attrs["name"] += "4"
        ds4_big.attrs["resolution"] = 500
        ds4_big.attrs["rows_per_scan"] = 1
        ds4_big.attrs["area"] = AreaDefinition("test", "test", "test",
                                               {"proj": "merc"}, 4, 4,
                                               (-2000, -2000, 2000, 2000))
        self.ds4_big = ds4_big

    @pytest.mark.parametrize(
        "init_kwargs",
        [
            {"high_resolution_band": "bad", "neutral_resolution_band": "red"},
            {"high_resolution_band": "red", "neutral_resolution_band": "bad"}
        ]
    )
    def test_bad_colors(self, init_kwargs):
        """Test that only valid band colors can be provided."""
        from satpy.composites import RatioSharpenedRGB
        with pytest.raises(ValueError, match="RatioSharpenedRGB..*_band must be one of .*"):
            RatioSharpenedRGB(name="true_color", **init_kwargs)

    def test_match_data_arrays(self):
        """Test that all areas have to be the same resolution."""
        from satpy.composites import IncompatibleAreas, RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color")
        with pytest.raises(IncompatibleAreas):
            comp((self.ds1, self.ds2, self.ds3), optional_datasets=(self.ds4_big,))

    def test_more_than_three_datasets(self):
        """Test that only 3 datasets can be passed."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color")
        with pytest.raises(ValueError, match="Expected 3 datasets, got 4"):
            comp((self.ds1, self.ds2, self.ds3, self.ds1), optional_datasets=(self.ds4_big,))

    def test_self_sharpened_no_high_res(self):
        """Test for exception when no high_res band is specified."""
        from satpy.composites import SelfSharpenedRGB
        comp = SelfSharpenedRGB(name="true_color", high_resolution_band=None)
        with pytest.raises(ValueError, match="SelfSharpenedRGB requires at least one high resolution band, not 'None'"):
            comp((self.ds1, self.ds2, self.ds3))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_no_high_res(self, dtype):
        """Test that three datasets can be passed without optional high res."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color")
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)))
        assert res.shape == (3, 2, 2)
        assert res.dtype == dtype
        assert res.values.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_no_sharpen(self, dtype):
        """Test that color None does no sharpening."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color", high_resolution_band=None)
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)),
                   optional_datasets=(self.ds4.astype(dtype),))
        assert res.shape == (3, 2, 2)
        assert res.dtype == dtype
        assert res.values.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        ("high_resolution_band", "neutral_resolution_band", "exp_r", "exp_g", "exp_b"),
        [
            ("red", None,
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[0.6, 0.6], [np.nan, 3.0]], dtype=np.float64),
             np.array([[0.8, 0.8], [np.nan, 4.0]], dtype=np.float64)),
            ("red", "green",
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[3.0, 3.0], [np.nan, 3.0]], dtype=np.float64),
             np.array([[0.8, 0.8], [np.nan, 4.0]], dtype=np.float64)),
            ("green", None,
             np.array([[5 / 3, 5 / 3], [np.nan, 0.0]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[4 / 3, 4 / 3], [np.nan, 4 / 3]], dtype=np.float64)),
            ("green", "blue",
             np.array([[5 / 3, 5 / 3], [np.nan, 0.0]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[4.0, 4.0], [np.nan, 4.0]], dtype=np.float64)),
            ("blue", None,
             np.array([[1.25, 1.25], [np.nan, 0.0]], dtype=np.float64),
             np.array([[0.75, 0.75], [np.nan, 0.75]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64)),
            ("blue", "red",
             np.array([[5.0, 5.0], [np.nan, 0.0]], dtype=np.float64),
             np.array([[0.75, 0.75], [np.nan, 0.75]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64))
        ]
    )
    def test_ratio_sharpening(self, high_resolution_band, neutral_resolution_band, exp_r, exp_g, exp_b, dtype):
        """Test RatioSharpenedRGB by different groups of high_resolution_band and neutral_resolution_band."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color", high_resolution_band=high_resolution_band,
                                 neutral_resolution_band=neutral_resolution_band)
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)),
                   optional_datasets=(self.ds4.astype(dtype),))

        assert "units" not in res.attrs
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.dtype == dtype

        data = res.values
        np.testing.assert_allclose(data[0], exp_r, rtol=1e-5)
        np.testing.assert_allclose(data[1], exp_g, rtol=1e-5)
        np.testing.assert_allclose(data[2], exp_b, rtol=1e-5)
        assert res.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        ("exp_shape", "exp_r", "exp_g", "exp_b"),
        [
            ((3, 2, 2),
             np.array([[5.0, 5.0], [5.0, 0]], dtype=np.float64),
             np.array([[4.0, 4.0], [4.0, 0]], dtype=np.float64),
             np.array([[16 / 3, 16 / 3], [16 / 3, 0]], dtype=np.float64))
        ]
    )
    def test_self_sharpened_basic(self, exp_shape, exp_r, exp_g, exp_b, dtype):
        """Test that three datasets can be passed without optional high res."""
        from satpy.composites import SelfSharpenedRGB
        comp = SelfSharpenedRGB(name="true_color")
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)))
        assert res.dtype == dtype

        data = res.values
        assert data.shape == exp_shape
        np.testing.assert_allclose(data[0], exp_r, rtol=1e-5)
        np.testing.assert_allclose(data[1], exp_g, rtol=1e-5)
        np.testing.assert_allclose(data[2], exp_b, rtol=1e-5)
        assert data.dtype == dtype


class TestDifferenceCompositor(unittest.TestCase):
    """Test case for the difference compositor."""

    def setUp(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition("test", "test", "test",
                              {"proj": "merc"}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        attrs = {"area": area,
                 "start_time": dt.datetime(2018, 1, 1, 18),
                 "modifiers": tuple(),
                 "resolution": 1000,
                 "name": "test_vis"}
        ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        self.ds1 = ds1
        ds2 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 2,
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds2.attrs["name"] += "2"
        self.ds2 = ds2

        # high res version
        ds2 = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64) + 4,
                           attrs=attrs.copy(), dims=("y", "x"),
                           coords={"y": [0, 1, 2, 3], "x": [0, 1, 2, 3]})
        ds2.attrs["name"] += "2"
        ds2.attrs["resolution"] = 500
        ds2.attrs["rows_per_scan"] = 1
        ds2.attrs["area"] = AreaDefinition("test", "test", "test",
                                           {"proj": "merc"}, 4, 4,
                                           (-2000, -2000, 2000, 2000))
        self.ds2_big = ds2

    def test_basic_diff(self):
        """Test that a basic difference composite works."""
        from satpy.composites import DifferenceCompositor
        comp = DifferenceCompositor(name="diff", standard_name="temperature_difference")
        res = comp((self.ds1, self.ds2))
        np.testing.assert_allclose(res.values, -2)
        assert res.attrs.get("standard_name") == "temperature_difference"

    def test_bad_areas_diff(self):
        """Test that a difference where resolutions are different fails."""
        from satpy.composites import DifferenceCompositor, IncompatibleAreas
        comp = DifferenceCompositor(name="diff")
        # too many arguments
        with pytest.raises(ValueError, match="Expected 2 datasets, got 3"):
            comp((self.ds1, self.ds2, self.ds2_big))
        # different resolution
        with pytest.raises(IncompatibleAreas):
            comp((self.ds1, self.ds2_big))


@pytest.fixture
def fake_area():
    """Return a fake 2×2 area."""
    from pyresample.geometry import create_area_def
    return create_area_def("skierffe", 4087, area_extent=[-5_000, -5_000, 5_000, 5_000], shape=(2, 2))


@pytest.fixture
def fake_dataset_pair(fake_area):
    """Return a fake pair of 2×2 datasets."""
    ds1 = xr.DataArray(da.full((2, 2), 8, chunks=2, dtype=np.float32),
                       attrs={"area": fake_area, "standard_name": "toa_bidirectional_reflectance"})
    ds2 = xr.DataArray(da.full((2, 2), 4, chunks=2, dtype=np.float32),
                       attrs={"area": fake_area, "standard_name": "toa_bidirectional_reflectance"})
    return (ds1, ds2)


@pytest.mark.parametrize("kwargs", [{}, {"standard_name": "channel_ratio", "foo": "bar"}])
def test_ratio_compositor(fake_dataset_pair, kwargs):
    """Test the ratio compositor."""
    from satpy.composites import RatioCompositor
    comp = RatioCompositor("ratio", **kwargs)
    res = comp(fake_dataset_pair)
    np.testing.assert_allclose(res.values, 2)

    assert res.attrs["name"] == "ratio"

    if "standard_name" in kwargs:
        # See that the kwargs have been updated to the attrs
        assert res.attrs["standard_name"] == "channel_ratio"
        assert res.attrs["foo"] == "bar"
    else:
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"


def test_sum_compositor(fake_dataset_pair):
    """Test the sum compositor."""
    from satpy.composites import SumCompositor
    comp = SumCompositor(name="sum", standard_name="channel_sum")
    res = comp(fake_dataset_pair)
    np.testing.assert_allclose(res.values, 12)


class TestDayNightCompositor(unittest.TestCase):
    """Test DayNightCompositor."""

    def setUp(self):
        """Create test data."""
        bands = ["R", "G", "B"]
        start_time = dt.datetime(2018, 1, 1, 18, 0, 0)

        # RGB
        a = np.zeros((3, 2, 2), dtype=np.float32)
        a[:, 0, 0] = 0.1
        a[:, 0, 1] = 0.2
        a[:, 1, 0] = 0.3
        a[:, 1, 1] = 0.4
        a = da.from_array(a, a.shape)
        self.data_a = xr.DataArray(a, attrs={"test": "a", "start_time": start_time},
                                   coords={"bands": bands}, dims=("bands", "y", "x"))
        b = np.zeros((3, 2, 2), dtype=np.float32)
        b[:, 0, 0] = np.nan
        b[:, 0, 1] = 0.25
        b[:, 1, 0] = 0.50
        b[:, 1, 1] = 0.75
        b = da.from_array(b, b.shape)
        self.data_b = xr.DataArray(b, attrs={"test": "b", "start_time": start_time},
                                   coords={"bands": bands}, dims=("bands", "y", "x"))

        sza = np.array([[80., 86.], [94., 100.]], dtype=np.float32)
        sza = da.from_array(sza, sza.shape)
        self.sza = xr.DataArray(sza, dims=("y", "x"))

        # fake area
        my_area = AreaDefinition(
            "test", "", "",
            "+proj=longlat",
            2, 2,
            (-95.0, 40.0, -92.0, 43.0),
        )
        self.data_a.attrs["area"] = my_area
        self.data_b.attrs["area"] = my_area
        # not used except to check that it matches the data arrays
        self.sza.attrs["area"] = my_area

    def test_daynight_sza(self):
        """Test compositor with both day and night portions when SZA data is included."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_night")
            res = comp((self.data_a, self.data_b, self.sza))
            res = res.compute()
        expected = np.array([[0., 0.22122374], [0.5, 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected, rtol=1e-6)

    def test_daynight_area(self):
        """Test compositor both day and night portions when SZA data is not provided."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_night")
            res = comp((self.data_a, self.data_b))
            res = res.compute()
        expected_channel = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        for i in range(3):
            np.testing.assert_allclose(res.values[i], expected_channel)

    def test_night_only_sza_with_alpha(self):
        """Test compositor with night portion with alpha band when SZA data is included."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=True)
            res = comp((self.data_b, self.sza))
            res = res.compute()
        expected_red_channel = np.array([[np.nan, 0.], [0.5, 1.]], dtype=np.float32)
        expected_alpha = np.array([[0., 0.3329599], [1., 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_red_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_night_only_sza_without_alpha(self):
        """Test compositor with night portion without alpha band when SZA data is included."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=False)
            res = comp((self.data_a, self.sza))
            res = res.compute()
        expected = np.array([[0., 0.11042609], [0.6683502, 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected)
        assert "A" not in res.bands

    def test_night_only_area_with_alpha(self):
        """Test compositor with night portion with alpha band when SZA data is not provided."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=True)
            res = comp((self.data_b,))
            res = res.compute()
        expected_l_channel = np.array([[np.nan, 0.], [0.5, 1.]], dtype=np.float32)
        expected_alpha = np.array([[np.nan, 0.], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_l_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_night_only_area_without_alpha(self):
        """Test compositor with night portion without alpha band when SZA data is not provided."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=False)
            res = comp((self.data_b,))
            res = res.compute()
        expected = np.array([[np.nan, 0.], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected)
        assert "A" not in res.bands

    def test_day_only_sza_with_alpha(self):
        """Test compositor with day portion with alpha band when SZA data is included."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=True)
            res = comp((self.data_a, self.sza))
            res = res.compute()
        expected_red_channel = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        expected_alpha = np.array([[1., 0.6670401], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_red_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_day_only_sza_without_alpha(self):
        """Test compositor with day portion without alpha band when SZA data is included."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=False)
            res = comp((self.data_a, self.sza))
            res = res.compute()
        expected_channel_data = np.array([[0., 0.22122373], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        for i in range(3):
            np.testing.assert_allclose(res.values[i], expected_channel_data)
        assert "A" not in res.bands

    def test_day_only_area_with_alpha(self):
        """Test compositor with day portion with alpha_band when SZA data is not provided."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=True)
            res = comp((self.data_a,))
            res = res.compute()
        expected_l_channel = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        expected_alpha = np.array([[1., 1.], [1., 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_l_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_day_only_area_with_alpha_and_missing_data(self):
        """Test compositor with day portion with alpha_band when SZA data is not provided and there is missing data."""
        from satpy.composites import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=True)
            res = comp((self.data_b,))
            res = res.compute()
        expected_l_channel = np.array([[np.nan, 0.], [0.5, 1.]], dtype=np.float32)
        expected_alpha = np.array([[np.nan, 1.], [1., 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_l_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_day_only_area_without_alpha(self):
        """Test compositor with day portion without alpha_band when SZA data is not provided."""
        from satpy.composites import DayNightCompositor

        # with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
        comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=False)
        res_dask = comp((self.data_a,))
        res = res_dask.compute()
        expected = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        assert res_dask.dtype == res.dtype
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected)
        assert "A" not in res.bands


class TestFillingCompositor(unittest.TestCase):
    """Test case for the filling compositor."""

    def test_fill(self):
        """Test filling."""
        from satpy.composites import FillingCompositor
        comp = FillingCompositor(name="fill_test")
        filler = xr.DataArray(np.array([1, 2, 3, 4, 3, 2, 1]))
        red = xr.DataArray(np.array([1, 2, 3, np.nan, 3, 2, 1]))
        green = xr.DataArray(np.array([np.nan, 2, 3, 4, 3, 2, np.nan]))
        blue = xr.DataArray(np.array([4, 3, 2, 1, 2, 3, 4]))
        res = comp([filler, red, green, blue])
        np.testing.assert_allclose(res.sel(bands="R").data, filler.data)
        np.testing.assert_allclose(res.sel(bands="G").data, filler.data)
        np.testing.assert_allclose(res.sel(bands="B").data, blue.data)


class TestMultiFiller(unittest.TestCase):
    """Test case for the MultiFiller compositor."""

    def test_fill(self):
        """Test filling."""
        from satpy.composites import MultiFiller
        comp = MultiFiller(name="fill_test")
        attrs = {"units": "K"}
        a = xr.DataArray(np.array([1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]), attrs=attrs.copy())
        b = xr.DataArray(np.array([np.nan, 2, 3, np.nan, np.nan, np.nan, np.nan]), attrs=attrs.copy())
        c = xr.DataArray(np.array([np.nan, 22, 3, np.nan, np.nan, np.nan, 7]), attrs=attrs.copy())
        d = xr.DataArray(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan]), attrs=attrs.copy())
        e = xr.DataArray(np.array([np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan]), attrs=attrs.copy())
        expected = xr.DataArray(np.array([1, 2, 3, np.nan, 5, 6, 7]))
        res = comp([a, b, c], optional_datasets=[d, e])
        np.testing.assert_allclose(res.data, expected.data)
        assert "units" in res.attrs
        assert res.attrs["units"] == "K"


class TestLuminanceSharpeningCompositor(unittest.TestCase):
    """Test luminance sharpening compositor."""

    def test_compositor(self):
        """Test luminance sharpening compositor."""
        from satpy.composites import LuminanceSharpeningCompositor
        comp = LuminanceSharpeningCompositor(name="test")
        # Three shades of grey
        rgb_arr = np.array([1, 50, 100, 200, 1, 50, 100, 200, 1, 50, 100, 200])
        rgb = xr.DataArray(rgb_arr.reshape((3, 2, 2)),
                           dims=["bands", "y", "x"], coords={"bands": ["R", "G", "B"]})
        # 100 % luminance -> all result values ~1.0
        lum = xr.DataArray(np.array([[100., 100.], [100., 100.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 1., atol=1e-9)
        # 50 % luminance, all result values ~0.5
        lum = xr.DataArray(np.array([[50., 50.], [50., 50.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.5, atol=1e-9)
        # 30 % luminance, all result values ~0.3
        lum = xr.DataArray(np.array([[30., 30.], [30., 30.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.3, atol=1e-9)
        # 0 % luminance, all values ~0.0
        lum = xr.DataArray(np.array([[0., 0.], [0., 0.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.0, atol=1e-9)


class TestSandwichCompositor:
    """Test sandwich compositor."""

    # Test RGB and RGBA
    @pytest.mark.parametrize(
        ("input_shape", "bands"),
        [
            ((3, 2, 2), ["R", "G", "B"]),
            ((4, 2, 2), ["R", "G", "B", "A"])
        ]
    )
    @mock.patch("satpy.composites.enhance2dataset")
    def test_compositor(self, e2d, input_shape, bands):
        """Test luminance sharpening compositor."""
        from satpy.composites import SandwichCompositor

        rgb_arr = da.from_array(RANDOM_GEN.random(input_shape), chunks=2)
        rgb = xr.DataArray(rgb_arr, dims=["bands", "y", "x"],
                           coords={"bands": bands})
        lum_arr = da.from_array(100 * RANDOM_GEN.random((2, 2)), chunks=2)
        lum = xr.DataArray(lum_arr, dims=["y", "x"])

        # Make enhance2dataset return unmodified dataset
        e2d.return_value = rgb
        comp = SandwichCompositor(name="test")

        res = comp([lum, rgb])

        for band in rgb:
            if band.bands != "A":
                # Check compositor has modified this band
                np.testing.assert_allclose(res.loc[band.bands].to_numpy(),
                                           band.to_numpy() * lum_arr / 100.)
            else:
                # Check Alpha band remains intact
                np.testing.assert_allclose(res.loc[band.bands].to_numpy(),
                                           band.to_numpy())
        # make sure the compositor doesn't modify the input data
        np.testing.assert_allclose(lum.values, lum_arr.compute())


class TestInlineComposites(unittest.TestCase):
    """Test inline composites."""

    def test_inline_composites(self):
        """Test that inline composites are working."""
        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        comps = load_compositor_configs_for_sensors(["visir"])[0]
        # Check that "fog" product has all its prerequisites defined
        keys = comps["visir"].keys()
        fog = [comps["visir"][dsid] for dsid in keys if "fog" == dsid["name"]][0]
        assert fog.attrs["prerequisites"][0]["name"] == "_fog_dep_0"
        assert fog.attrs["prerequisites"][1]["name"] == "_fog_dep_1"
        assert fog.attrs["prerequisites"][2] == 10.8

        # Check that the sub-composite dependencies use wavelengths
        # (numeric values)
        keys = comps["visir"].keys()
        fog_dep_ids = [dsid for dsid in keys if "fog_dep" in dsid["name"]]
        assert comps["visir"][fog_dep_ids[0]].attrs["prerequisites"] == [12.0, 10.8]
        assert comps["visir"][fog_dep_ids[1]].attrs["prerequisites"] == [10.8, 8.7]

        # Check the same for SEVIRI and verify channel names are used
        # in the sub-composite dependencies instead of wavelengths
        comps = load_compositor_configs_for_sensors(["seviri"])[0]
        keys = comps["seviri"].keys()
        fog_dep_ids = [dsid for dsid in keys if "fog_dep" in dsid["name"]]
        assert comps["seviri"][fog_dep_ids[0]].attrs["prerequisites"] == ["IR_120", "IR_108"]
        assert comps["seviri"][fog_dep_ids[1]].attrs["prerequisites"] == ["IR_108", "IR_087"]


class TestColormapCompositor(unittest.TestCase):
    """Test the ColormapCompositor."""

    def setUp(self):
        """Set up the test case."""
        from satpy.composites import ColormapCompositor
        self.colormap_compositor = ColormapCompositor("test_cmap_compositor")

    def test_build_colormap_with_int_data_and_without_meanings(self):
        """Test colormap building."""
        palette = np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]])
        colormap, squeezed_palette = self.colormap_compositor.build_colormap(palette, np.uint8, {})
        assert np.allclose(colormap.values, [0, 1])
        assert np.allclose(squeezed_palette, palette / 255.0)

    def test_build_colormap_with_int_data_and_with_meanings(self):
        """Test colormap building."""
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]
        colormap, squeezed_palette = self.colormap_compositor.build_colormap(palette, np.uint8, {})
        assert np.allclose(colormap.values, [2, 3, 4])
        assert np.allclose(squeezed_palette, palette / 255.0)


class TestPaletteCompositor(unittest.TestCase):
    """Test the PaletteCompositor."""

    def test_call(self):
        """Test palette compositing."""
        from satpy.composites import PaletteCompositor
        cmap_comp = PaletteCompositor("test_cmap_compositor")
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]

        data = xr.DataArray(da.from_array(np.array([[4, 3, 2], [2, 3, 4]], dtype=np.uint8)), dims=["y", "x"])
        res = cmap_comp([data, palette])
        exp = np.array([[[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]]])
        assert np.allclose(res, exp)


class TestColorizeCompositor(unittest.TestCase):
    """Test the ColorizeCompositor."""

    def test_colorize_no_fill(self):
        """Test colorizing."""
        from satpy.composites import ColorizeCompositor
        colormap_composite = ColorizeCompositor("test_color_compositor")
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]

        data = xr.DataArray(np.array([[4, 3, 2],
                                      [2, 3, 4]],
                                     dtype=np.uint8),
                            dims=["y", "x"])
        res = colormap_composite([data, palette])
        exp = np.array([[[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]]])
        assert np.allclose(res, exp, atol=0.0001)

    def test_colorize_with_interpolation(self):
        """Test colorizing with interpolation."""
        from satpy.composites import ColorizeCompositor
        colormap_composite = ColorizeCompositor("test_color_compositor")
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]

        data = xr.DataArray(da.from_array(np.array([[4, 3, 2.5],
                                                    [2, 3.2, 4]])),
                            dims=["y", "x"],
                            attrs={"valid_range": np.array([2, 4])})
        res = colormap_composite([data, palette])
        exp = np.array([[[1.0, 0.498039, 0.246575],
                         [0., 0.59309977, 1.0]],
                        [[1.0, 0.49803924, 0.24657543],
                         [0., 0.59309983, 1.0]],
                        [[1.0, 0.4980392, 0.24657541],
                         [0., 0.59309978, 1.0]]])
        np.testing.assert_allclose(res, exp, atol=1e-4)


class TestCloudCompositorWithoutCloudfree:
    """Test the CloudCompositorWithoutCloudfree."""

    def setup_method(self):
        """Set up the test case."""
        from satpy.composites.cloud_products import CloudCompositorWithoutCloudfree
        self.colormap_composite = CloudCompositorWithoutCloudfree("test_cmap_compositor")

        self.exp = np.array([[4, 3, 2], [2, 3, np.nan], [8, 7, 655350]])
        self.exp_bad_oc = np.array([[4, 3, 2],
                                    [2, np.nan, 4],
                                    [np.nan, 7, 255]])

    def test_call_numpy_with_invalid_value_in_status(self):
        """Test the CloudCompositorWithoutCloudfree composite generation."""
        status = xr.DataArray(np.array([[0, 0, 0], [0, 0, 65535], [0, 0, 1]]), dims=["y", "x"],
                              attrs={"_FillValue": 65535})
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, np.nan], [8, 7, np.nan]], dtype=np.float32),
                            dims=["y", "x"],
                            attrs={"_FillValue": 65535,
                                   "scaled_FillValue": 655350})
        res = self.colormap_composite([data, status])
        np.testing.assert_allclose(res, self.exp, atol=1e-4)

    def test_call_dask_with_invalid_value_in_status(self):
        """Test the CloudCompositorWithoutCloudfree composite generation."""
        status = xr.DataArray(da.from_array(np.array([[0, 0, 0], [0, 0, 65535], [0, 0, 1]])), dims=["y", "x"],
                              attrs={"_FillValue": 65535})
        data = xr.DataArray(da.from_array(np.array([[4, 3, 2], [2, 3, np.nan], [8, 7, np.nan]], dtype=np.float32)),
                            dims=["y", "x"],
                            attrs={"_FillValue": 99,
                                   "scaled_FillValue": 655350})
        res = self.colormap_composite([data, status])
        np.testing.assert_allclose(res, self.exp, atol=1e-4)

    def test_call_bad_optical_conditions(self):
        """Test the CloudCompositorWithoutCloudfree composite generation."""
        status = xr.DataArray(da.from_array(np.array([[0, 0, 0], [3, 3, 3], [0, 0, 1]])), dims=["y", "x"],
                              attrs={"_FillValue": 65535,
                                     "flag_meanings": "bad_optical_conditions"})
        data = xr.DataArray(np.array([[4, 3, 2], [2, 255, 4], [255, 7, 255]], dtype=np.uint8),
                            dims=["y", "x"],
                            name="cmic_cre",
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        res = self.colormap_composite([data, status])
        np.testing.assert_allclose(res, self.exp_bad_oc, atol=1e-4)

    def test_bad_indata(self):
        """Test the CloudCompositorWithoutCloudfree composite generation without status."""
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, 4], [255, 7, 255]], dtype=np.uint8),
                            dims=["y", "x"],
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        np.testing.assert_raises(ValueError, self.colormap_composite, [data])


class TestCloudCompositorCommonMask:
    """Test the CloudCompositorCommonMask."""

    def setup_method(self):
        """Set up the test case."""
        from satpy.composites.cloud_products import CloudCompositorCommonMask

        self.exp_a = np.array([[4, 3, 2],
                               [2, 3, 655350],
                               [np.nan, np.nan, np.nan]])
        self.exp_b = np.array([[4, 3, 2],
                               [2, 3, 255],
                               [np.nan, np.nan, np.nan]])
        self.colormap_composite = CloudCompositorCommonMask("test_cmap_compositor")

    def test_call_numpy(self):
        """Test the CloudCompositorCommonMask with numpy."""
        mask = xr.DataArray(np.array([[0, 0, 0], [1, 1, 1], [255, 255, 255]]), dims=["y", "x"],
                            attrs={"_FillValue": 255})
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32),
                            dims=["y", "x"],
                            attrs={"_FillValue": 65535,
                                   "scaled_FillValue": 655350})
        res = self.colormap_composite([data, mask])
        np.testing.assert_allclose(res, self.exp_a, atol=1e-4)

    def test_call_dask(self):
        """Test the CloudCompositorCommonMask with dask."""
        mask = xr.DataArray(da.from_array(np.array([[0, 0, 0], [1, 1, 1], [255, 255, 255]])), dims=["y", "x"],
                            attrs={"_FillValue": 255})
        data = xr.DataArray(da.from_array(np.array([[4, 3, 2], [2, 3, 255], [255, 255, 255]], dtype=np.int16)),
                            dims=["y", "x"],
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        res = self.colormap_composite([data, mask])
        np.testing.assert_allclose(res, self.exp_b, atol=1e-4)

    def test_bad_call(self):
        """Test the CloudCompositorCommonMask without mask."""
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, 255], [255, 255, 255]], dtype=np.int16),
                            dims=["y", "x"],
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        np.testing.assert_raises(ValueError, self.colormap_composite, [data])


class TestPrecipCloudsCompositor(unittest.TestCase):
    """Test the PrecipClouds compositor."""

    def test_call(self):
        """Test the precip composite generation."""
        from satpy.composites.cloud_products import PrecipCloudsRGB
        colormap_compositor = PrecipCloudsRGB("test_precip_compositor")

        data_light = xr.DataArray(np.array([[80, 70, 60, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                  dims=["y", "x"], attrs={"_FillValue": 255})
        data_moderate = xr.DataArray(np.array([[60, 50, 40, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                     dims=["y", "x"], attrs={"_FillValue": 255})
        data_intense = xr.DataArray(np.array([[40, 30, 20, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                    dims=["y", "x"], attrs={"_FillValue": 255})
        data_flags = xr.DataArray(np.array([[0, 0, 4, 0], [0, 0, 0, 0]], dtype=np.uint8),
                                  dims=["y", "x"])
        res = colormap_compositor([data_light, data_moderate, data_intense, data_flags])

        exp = np.array([[[0.24313725, 0.18235294, 0.12156863, np.nan],
                         [0.12156863, 0.18235294, 0.24313725, np.nan]],
                        [[0.62184874, 0.51820728, 0.41456583, np.nan],
                         [0.20728291, 0.31092437, 0.41456583, np.nan]],
                        [[0.82913165, 0.7254902, 0.62184874, np.nan],
                         [0.20728291, 0.31092437, 0.41456583, np.nan]]])

        np.testing.assert_allclose(res, exp)


class TestHighCloudCompositor:
    """Test HighCloudCompositor."""

    def setup_method(self):
        """Create test data."""
        from pyresample.geometry import create_area_def
        area = create_area_def(area_id="test", projection={"proj": "latlong"},
                               center=(0, 45), width=3, height=3, resolution=35)
        self.dtype = np.float32
        self.data = xr.DataArray(
            da.from_array(np.array([[200, 250, 300], [200, 250, 300], [200, 250, 300]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]},
            attrs={"area": area}
        )

    def test_high_cloud_compositor(self):
        """Test general default functionality of compositor."""
        from satpy.composites import HighCloudCompositor
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = HighCloudCompositor(name="test")
            res = comp([self.data])
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        expexted_alpha = np.array([[1.0, 0.7142857, 0.0], [1.0, 0.625, 0.0], [1.0, 0.5555555, 0.0]])
        expected = np.stack([self.data, expexted_alpha])
        np.testing.assert_almost_equal(res.values, expected)

    def test_high_cloud_compositor_multiple_calls(self):
        """Test that the modified init variables are reset properly when calling the compositor multiple times."""
        from satpy.composites import HighCloudCompositor
        comp = HighCloudCompositor(name="test")
        res = comp([self.data])
        res2 = comp([self.data])
        np.testing.assert_equal(res.values, res2.values)

    def test_high_cloud_compositor_dtype(self):
        """Test that the datatype is not altered by the compositor."""
        from satpy.composites import HighCloudCompositor
        comp = HighCloudCompositor(name="test")
        res = comp([self.data])
        assert res.data.dtype == self.dtype

    def test_high_cloud_compositor_validity_checks(self):
        """Test that errors are raised for invalid input data and settings."""
        from satpy.composites import HighCloudCompositor

        with pytest.raises(ValueError, match="Expected 2 `transition_min_limits` values, got 1"):
            _ = HighCloudCompositor("test", transition_min_limits=(210., ))

        with pytest.raises(ValueError, match="Expected 2 `latitude_min_limits` values, got 3"):
            _ = HighCloudCompositor("test", latitude_min_limits=(20., 40., 60.))

        with pytest.raises(ValueError, match="Expected `transition_max` to be of type float, "
                                             "is of type <class 'tuple'>"):
            _ = HighCloudCompositor("test", transition_max=(250., 300.))

        comp = HighCloudCompositor("test")
        with pytest.raises(ValueError, match="Expected 1 dataset, got 2"):
            _ = comp([self.data, self.data])


class TestLowCloudCompositor:
    """Test LowCloudCompositor."""

    def setup_method(self):
        """Create test data."""
        self.dtype = np.float32
        self.btd = xr.DataArray(
            da.from_array(np.array([[0.0, 1.0, 10.0], [0.0, 1.0, 10.0], [0.0, 1.0, 10.0]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]}
        )
        self.bt_win = xr.DataArray(
            da.from_array(np.array([[250, 250, 250], [250, 250, 250], [150, 150, 150]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]}
        )
        self.lsm = xr.DataArray(
            da.from_array(np.array([[0., 0., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]}
        )

    def test_low_cloud_compositor(self):
        """Test general default functionality of compositor."""
        from satpy.composites import LowCloudCompositor
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = LowCloudCompositor(name="test")
            res = comp([self.btd, self.bt_win, self.lsm])
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        expexted_alpha = np.array([[0.0, 0.25, 1.0], [0.0, 0.25, 1.0], [0.0, 0.0, 0.0]])
        expected = np.stack([self.btd, expexted_alpha])
        np.testing.assert_equal(res.values, expected)

    def test_low_cloud_compositor_dtype(self):
        """Test that the datatype is not altered by the compositor."""
        from satpy.composites import LowCloudCompositor
        comp = LowCloudCompositor(name="test")
        res = comp([self.btd, self.bt_win, self.lsm])
        assert res.data.dtype == self.dtype

    def test_low_cloud_compositor_validity_checks(self):
        """Test that errors are raised for invalid input data and settings."""
        from satpy.composites import LowCloudCompositor

        with pytest.raises(ValueError, match="Expected 2 `range_land` values, got 1"):
            _ = LowCloudCompositor("test", range_land=(2.0, ))

        with pytest.raises(ValueError, match="Expected 2 `range_water` values, got 1"):
            _ = LowCloudCompositor("test", range_water=(2.0,))

        comp = LowCloudCompositor("test")
        with pytest.raises(ValueError, match="Expected 3 datasets, got 2"):
            _ = comp([self.btd, self.lsm])


class TestSingleBandCompositor(unittest.TestCase):
    """Test the single-band compositor."""

    def setUp(self):
        """Create test data."""
        from satpy.composites import SingleBandCompositor
        self.comp = SingleBandCompositor(name="test")

        all_valid = np.ones((2, 2))
        self.all_valid = xr.DataArray(all_valid, dims=["y", "x"])

    def test_call(self):
        """Test calling the compositor."""
        # Dataset with extra attributes
        all_valid = self.all_valid
        all_valid.attrs["sensor"] = "foo"
        attrs = {
            "foo": "bar",
            "resolution": 333,
            "units": "K",
            "sensor": {"fake_sensor1", "fake_sensor2"},
            "calibration": "BT",
            "wavelength": 10.8
        }
        self.comp.attrs["resolution"] = None
        res = self.comp([all_valid], **attrs)
        # Verify attributes
        assert res.attrs.get("sensor") == "foo"
        assert "foo" in res.attrs
        assert res.attrs.get("foo") == "bar"
        assert "units" in res.attrs
        assert "calibration" in res.attrs
        assert "modifiers" not in res.attrs
        assert res.attrs["wavelength"] == 10.8
        assert res.attrs["resolution"] == 333


class TestCategoricalDataCompositor(unittest.TestCase):
    """Test composiotor for recategorization of categorical data."""

    def setUp(self):
        """Create test data."""
        attrs = {"name": "foo"}
        data = xr.DataArray(da.from_array([[2., 1.], [3., 0.]]), attrs=attrs,
                            dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})

        self.data = data

    def test_basic_recategorization(self):
        """Test general functionality of compositor incl. attributes."""
        from satpy.composites import CategoricalDataCompositor
        lut = [np.nan, 0, 1, 1]
        name = "bar"
        comp = CategoricalDataCompositor(name=name, lut=lut)
        res = comp([self.data])
        res = res.compute()
        expected = np.array([[1., 0.], [1., np.nan]])
        np.testing.assert_equal(res.values, expected)
        np.testing.assert_equal(res.attrs["name"], name)
        np.testing.assert_equal(res.attrs["composite_lut"], lut)

    def test_too_many_datasets(self):
        """Test that ValueError is raised if more than one dataset is provided."""
        from satpy.composites import CategoricalDataCompositor
        lut = [np.nan, 0, 1, 1]
        comp = CategoricalDataCompositor(name="foo", lut=lut)
        np.testing.assert_raises(ValueError, comp, [self.data, self.data])


class TestGenericCompositor(unittest.TestCase):
    """Test generic compositor."""

    def setUp(self):
        """Create test data."""
        from satpy.composites import GenericCompositor
        self.comp = GenericCompositor(name="test")
        self.comp2 = GenericCompositor(name="test2", common_channel_mask=False)

        all_valid = np.ones((1, 2, 2))
        self.all_valid = xr.DataArray(all_valid, dims=["bands", "y", "x"])
        first_invalid = np.reshape(np.array([np.nan, 1., 1., 1.]), (1, 2, 2))
        self.first_invalid = xr.DataArray(first_invalid,
                                          dims=["bands", "y", "x"])
        second_invalid = np.reshape(np.array([1., np.nan, 1., 1.]), (1, 2, 2))
        self.second_invalid = xr.DataArray(second_invalid,
                                           dims=["bands", "y", "x"])
        wrong_shape = np.reshape(np.array([1., 1., 1.]), (1, 3, 1))
        self.wrong_shape = xr.DataArray(wrong_shape, dims=["bands", "y", "x"])

    def test_masking(self):
        """Test masking in generic compositor."""
        # Single channel
        res = self.comp([self.all_valid])
        np.testing.assert_allclose(res.data, 1., atol=1e-9)
        # Three channels, one value invalid
        res = self.comp([self.all_valid, self.all_valid, self.first_invalid])
        correct = np.reshape(np.array([np.nan, 1., 1., 1.]), (2, 2))
        for i in range(3):
            np.testing.assert_almost_equal(res.data[i, :, :], correct)
        # Three channels, two values invalid
        res = self.comp([self.all_valid, self.first_invalid, self.second_invalid])
        correct = np.reshape(np.array([np.nan, np.nan, 1., 1.]), (2, 2))
        for i in range(3):
            np.testing.assert_almost_equal(res.data[i, :, :], correct)

    def test_concat_datasets(self):
        """Test concatenation of datasets."""
        from satpy.composites import IncompatibleAreas
        res = self.comp._concat_datasets([self.all_valid], "L")
        num_bands = len(res.bands)
        assert num_bands == 1
        assert res.shape[0] == num_bands
        assert res.bands[0] == "L"
        res = self.comp._concat_datasets([self.all_valid, self.all_valid], "LA")
        num_bands = len(res.bands)
        assert num_bands == 2
        assert res.shape[0] == num_bands
        assert res.bands[0] == "L"
        assert res.bands[1] == "A"
        with pytest.raises(IncompatibleAreas):
            self.comp._concat_datasets([self.all_valid, self.wrong_shape], "LA")

    def test_get_sensors(self):
        """Test getting sensors from the dataset attributes."""
        res = self.comp._get_sensors([self.all_valid])
        assert res is None
        dset1 = self.all_valid
        dset1.attrs["sensor"] = "foo"
        res = self.comp._get_sensors([dset1])
        assert res == "foo"
        dset2 = self.first_invalid
        dset2.attrs["sensor"] = "bar"
        res = self.comp._get_sensors([dset1, dset2])
        assert "foo" in res
        assert "bar" in res
        assert len(res) == 2
        assert isinstance(res, set)

    @mock.patch("satpy.composites.GenericCompositor._get_sensors")
    @mock.patch("satpy.composites.combine_metadata")
    @mock.patch("satpy.composites.check_times")
    @mock.patch("satpy.composites.GenericCompositor.match_data_arrays")
    def test_call_with_mock(self, match_data_arrays, check_times, combine_metadata, get_sensors):
        """Test calling generic compositor."""
        from satpy.composites import IncompatibleAreas
        combine_metadata.return_value = dict()
        get_sensors.return_value = "foo"
        # One dataset, no mode given
        res = self.comp([self.all_valid])
        assert res.shape[0] == 1
        assert res.attrs["mode"] == "L"
        match_data_arrays.assert_not_called()
        # This compositor has been initialized without common masking, so the
        # masking shouldn't have been called
        projectables = [self.all_valid, self.first_invalid, self.second_invalid]
        match_data_arrays.return_value = projectables
        res = self.comp2(projectables)
        match_data_arrays.assert_called_once()
        match_data_arrays.reset_mock()
        # Dataset for alpha given, so shouldn't be masked
        projectables = [self.all_valid, self.all_valid]
        match_data_arrays.return_value = projectables
        res = self.comp(projectables)
        match_data_arrays.assert_called_once()
        match_data_arrays.reset_mock()
        # When areas are incompatible, masking shouldn't happen
        match_data_arrays.side_effect = IncompatibleAreas()
        with pytest.raises(IncompatibleAreas):
            self.comp([self.all_valid, self.wrong_shape])
        match_data_arrays.assert_called_once()

    def test_call(self):
        """Test calling generic compositor."""
        # Multiple datasets with extra attributes
        all_valid = self.all_valid
        all_valid.attrs["sensor"] = "foo"
        attrs = {"foo": "bar", "resolution": 333}
        self.comp.attrs["resolution"] = None
        res = self.comp([self.all_valid, self.first_invalid], **attrs)
        # Verify attributes
        assert res.attrs.get("sensor") == "foo"
        assert "foo" in res.attrs
        assert res.attrs.get("foo") == "bar"
        assert "units" not in res.attrs
        assert "calibration" not in res.attrs
        assert "modifiers" not in res.attrs
        assert res.attrs["wavelength"] is None
        assert res.attrs["mode"] == "LA"
        assert res.attrs["resolution"] == 333

    def test_deprecation_warning(self):
        """Test deprecation warning for dcprecated composite recipes."""
        warning_message = "foo is a deprecated composite. Use composite bar instead."
        self.comp.attrs["deprecation_warning"] = warning_message
        with pytest.warns(UserWarning, match=warning_message):
            self.comp([self.all_valid])


class TestAddBands(unittest.TestCase):
    """Test case for the `add_bands` function."""

    def test_add_bands_l_rgb(self):
        """Test adding bands."""
        from satpy.composites import add_bands

        # L + RGB -> RGB
        data = xr.DataArray(da.ones((1, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["L"]})
        new_bands = xr.DataArray(da.array(["R", "G", "B"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B"]
        assert res.attrs["mode"] == "".join(res_bands)
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords["bands"], res_bands)
        assert res.dtype == np.float32

    def test_add_bands_l_rgba(self):
        """Test adding bands."""
        from satpy.composites import add_bands

        # L + RGBA -> RGBA
        data = xr.DataArray(da.ones((1, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["L"]}, attrs={"mode": "L"})
        new_bands = xr.DataArray(da.array(["R", "G", "B", "A"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B", "A"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B", "A"]
        assert res.attrs["mode"] == "".join(res_bands)
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords["bands"], res_bands)
        assert res.dtype == np.float32

    def test_add_bands_la_rgb(self):
        """Test adding bands."""
        from satpy.composites import add_bands

        # LA + RGB -> RGBA
        data = xr.DataArray(da.ones((2, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["L", "A"]}, attrs={"mode": "LA"})
        new_bands = xr.DataArray(da.array(["R", "G", "B"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B", "A"]
        assert res.attrs["mode"] == "".join(res_bands)
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords["bands"], res_bands)
        assert res.dtype == np.float32

    def test_add_bands_rgb_rbga(self):
        """Test adding bands."""
        from satpy.composites import add_bands

        # RGB + RGBA -> RGBA
        data = xr.DataArray(da.ones((3, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["R", "G", "B"]},
                            attrs={"mode": "RGB"})
        new_bands = xr.DataArray(da.array(["R", "G", "B", "A"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B", "A"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B", "A"]
        assert res.attrs["mode"] == "".join(res_bands)
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords["bands"], res_bands)
        assert res.dtype == np.float32

    def test_add_bands_p_l(self):
        """Test adding bands."""
        from satpy.composites import add_bands

        # P(RGBA) + L -> RGBA
        data = xr.DataArray(da.ones((1, 3, 3)), dims=("bands", "y", "x"),
                            coords={"bands": ["P"]},
                            attrs={"mode": "P"})
        new_bands = xr.DataArray(da.array(["L"]), dims=("bands"),
                                 coords={"bands": ["L"]})
        with pytest.raises(NotImplementedError):
            add_bands(data, new_bands)


class TestStaticImageCompositor(unittest.TestCase):
    """Test case for the static compositor."""

    @mock.patch("satpy.area.get_area_def")
    def test_init(self, get_area_def):
        """Test the initializiation of static compositor."""
        from satpy.composites import StaticImageCompositor

        # No filename given raises ValueError
        with pytest.raises(ValueError, match="StaticImageCompositor needs a .*"):
            StaticImageCompositor("name")

        # No area defined
        comp = StaticImageCompositor("name", filename="/foo.tif")
        assert comp._cache_filename == "/foo.tif"
        assert comp.area is None

        # Area defined
        get_area_def.return_value = "bar"
        comp = StaticImageCompositor("name", filename="/foo.tif", area="euro4")
        assert comp._cache_filename == "/foo.tif"
        assert comp.area == "bar"
        get_area_def.assert_called_once_with("euro4")

    @mock.patch("satpy.aux_download.retrieve")
    @mock.patch("satpy.aux_download.register_file")
    @mock.patch("satpy.Scene")
    def test_call(self, Scene, register, retrieve):  # noqa
        """Test the static compositing."""
        from satpy.composites import StaticImageCompositor

        satpy.config.set(data_dir=os.path.join(os.path.sep, "path", "to", "image"))
        remote_tif = "http://example.com/foo.tif"

        class MockScene(dict):
            def load(self, arg):
                pass

        img = mock.MagicMock()
        img.attrs = {}
        scn = MockScene()
        scn["image"] = img
        Scene.return_value = scn
        # absolute path to local file
        comp = StaticImageCompositor("name", filename="/foo.tif", area="euro4")
        res = comp()
        Scene.assert_called_once_with(reader="generic_image",
                                      filenames=["/foo.tif"])
        register.assert_not_called()
        retrieve.assert_not_called()
        assert res.attrs["sensor"] is None
        assert "modifiers" not in res.attrs
        assert "calibration" not in res.attrs

        # remote file with local cached version
        Scene.reset_mock()
        register.return_value = "data_dir/foo.tif"
        retrieve.return_value = "data_dir/foo.tif"
        comp = StaticImageCompositor("name", url=remote_tif, area="euro4")
        res = comp()
        Scene.assert_called_once_with(reader="generic_image",
                                      filenames=["data_dir/foo.tif"])
        assert res.attrs["sensor"] is None
        assert "modifiers" not in res.attrs
        assert "calibration" not in res.attrs

        # Non-georeferenced image, no area given
        img.attrs.pop("area")
        comp = StaticImageCompositor("name", filename="/foo.tif")
        with pytest.raises(AttributeError):
            comp()

        # Non-georeferenced image, area given
        comp = StaticImageCompositor("name", filename="/foo.tif", area="euro4")
        res = comp()
        assert res.attrs["area"].area_id == "euro4"

        # Filename contains environment variable
        os.environ["TEST_IMAGE_PATH"] = "/path/to/image"
        comp = StaticImageCompositor("name", filename="${TEST_IMAGE_PATH}/foo.tif", area="euro4")
        assert comp._cache_filename == "/path/to/image/foo.tif"

        # URL and filename without absolute path
        comp = StaticImageCompositor("name", url=remote_tif, filename="bar.tif")
        assert comp._url == remote_tif
        assert comp._cache_filename == "bar.tif"

        # No URL, filename without absolute path, use default data_dir from config
        with mock.patch("os.path.exists") as exists:
            exists.return_value = True
            comp = StaticImageCompositor("name", filename="foo.tif")
            assert comp._url is None
            assert comp._cache_filename == os.path.join(os.path.sep, "path", "to", "image", "foo.tif")


def _enhance2dataset(dataset, convert_p=False):
    """Mock the enhance2dataset to return the original data."""
    return dataset


class TestBackgroundCompositor:
    """Test case for the background compositor."""

    @classmethod
    def setup_class(cls):
        """Create shared input data arrays."""
        foreground_data = {
            "L": np.array([[[1., 0.5], [0., np.nan]]]),
            "LA": np.array([[[1., 0.5], [0., np.nan]], [[0.5, 0.5], [0.5, 0.5]]]),
            "RGB": np.array([
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]]]),
            "RGBA": np.array([
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]],
                [[0.5, 0.5], [0., 0.5]]]),
        }
        cls.foreground_data = foreground_data

    @mock.patch("satpy.composites.enhance2dataset", _enhance2dataset)
    @pytest.mark.parametrize(
        ("foreground_bands", "background_bands", "exp_bands", "exp_result"),
        [
            ("L", "L", "L", np.array([[1., 0.5], [0., 1.]])),
            ("L", "RGB", "RGB", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]]])),
            ("LA", "LA", "LA", np.array([
                [[1., 0.75], [0.5, 1.]],
                [[1., 1.], [1., 1.]]])),
            ("LA", "RGB", "RGB", np.array([
                [[1., 0.75], [0.5, 1.]],
                [[1., 0.75], [0.5, 1.]],
                [[1., 0.75], [0.5, 1.]]])),
            ("RGB", "RGB", "RGB", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]]])),
            ("RGB", "LA", "RGBA", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 1.], [1., 1.]]])),
            ("RGB", "RGBA", "RGBA", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 1.], [1., 1.]]])),
            ("RGBA", "RGBA", "RGBA", np.array([
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]],
                [[1., 1.], [1., 1.]]])),
            ("RGBA", "RGB", "RGB", np.array([
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]]])),
        ]
    )
    def test_call(self, foreground_bands, background_bands, exp_bands, exp_result):
        """Test the background compositing."""
        from satpy.composites import BackgroundCompositor
        comp = BackgroundCompositor("name")

        # L mode images
        foreground_data = self.foreground_data[foreground_bands]

        attrs = {"mode": foreground_bands, "area": "foo"}
        foreground = xr.DataArray(da.from_array(foreground_data),
                                  dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs)
        attrs = {"mode": background_bands, "area": "foo"}
        background = xr.DataArray(da.ones((len(background_bands), 2, 2)), dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs)

        res = comp([foreground, background])

        assert res.attrs["area"] == "foo"
        np.testing.assert_allclose(res, exp_result)
        assert res.attrs["mode"] == exp_bands

    @mock.patch("satpy.composites.enhance2dataset", _enhance2dataset)
    def test_multiple_sensors(self):
        """Test the background compositing from multiple sensor data."""
        from satpy.composites import BackgroundCompositor
        comp = BackgroundCompositor("name")

        # L mode images
        attrs = {"mode": "L", "area": "foo"}
        foreground_data = self.foreground_data["L"]
        foreground = xr.DataArray(da.from_array(foreground_data),
                                  dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs.copy())
        foreground.attrs["sensor"] = "abi"
        background = xr.DataArray(da.ones((1, 2, 2)), dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs.copy())
        background.attrs["sensor"] = "glm"
        res = comp([foreground, background])
        assert res.attrs["area"] == "foo"
        np.testing.assert_allclose(res, np.array([[1., 0.5], [0., 1.]]))
        assert res.attrs["mode"] == "L"
        assert res.attrs["sensor"] == {"abi", "glm"}


class TestMaskingCompositor:
    """Test case for the simple masking compositor."""

    @pytest.fixture
    def conditions_v1(self):
        """Masking conditions with string values."""
        return [{"method": "equal",
                 "value": "Cloud-free_land",
                 "transparency": 100},
                {"method": "equal",
                 "value": "Cloud-free_sea",
                 "transparency": 50}]

    @pytest.fixture
    def conditions_v2(self):
        """Masking conditions with numerical values."""
        return [{"method": "equal",
                 "value": 1,
                 "transparency": 100},
                {"method": "equal",
                 "value": 2,
                 "transparency": 50}]

    @pytest.fixture
    def conditions_v3(self):
        """Masking conditions with other numerical values."""
        return [{"method": "equal",
                 "value": 0,
                 "transparency": 100},
                {"method": "equal",
                 "value": 1,
                 "transparency": 0}]

    @pytest.fixture
    def test_data(self):
        """Test data to use with masking compositors."""
        return xr.DataArray(da.random.random((3, 3)), dims=["y", "x"])

    @pytest.fixture
    def test_ct_data(self):
        """Test 2D CT data array."""
        flag_meanings = ["Cloud-free_land", "Cloud-free_sea"]
        flag_values = da.array([1, 2])
        ct_data = da.array([[1, 2, 2],
                            [2, 1, 2],
                            [2, 2, 1]])
        ct_data = xr.DataArray(ct_data, dims=["y", "x"])
        ct_data.attrs["flag_meanings"] = flag_meanings
        ct_data.attrs["flag_values"] = flag_values
        return ct_data

    @pytest.fixture
    def value_3d_data(self):
        """Test 3D data array."""
        value_3d_data = da.array([[[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]]])
        value_3d_data = xr.DataArray(value_3d_data, dims=["bands", "y", "x"])
        return value_3d_data

    @pytest.fixture
    def value_3d_data_bands(self):
        """Test 3D data array."""
        value_3d_data = da.array([[[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]],
                                  [[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]],
                                  [[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]]])
        value_3d_data = xr.DataArray(value_3d_data, dims=["bands", "y", "x"])
        return value_3d_data

    @pytest.fixture
    def test_ct_data_v3(self, test_ct_data):
        """Set ct data to NaN where it originally is 1."""
        return test_ct_data.where(test_ct_data == 1)

    @pytest.fixture
    def reference_data(self, test_data, test_ct_data):
        """Get reference data to use in masking compositor tests."""
        # The data are set to NaN where ct is `1`
        return test_data.where(test_ct_data > 1)

    @pytest.fixture
    def reference_alpha(self):
        """Get reference alpha to use in masking compositor tests."""
        ref_alpha = da.array([[0, 0.5, 0.5],
                              [0.5, 0, 0.5],
                              [0.5, 0.5, 0]])
        return xr.DataArray(ref_alpha, dims=["y", "x"])

    def test_init(self):
        """Test the initializiation of compositor."""
        from satpy.composites import MaskingCompositor

        # No transparency or conditions given raises ValueError
        with pytest.raises(ValueError, match="Masking conditions not defined."):
            _ = MaskingCompositor("name")

        # transparency defined
        transparency = {0: 100, 1: 50}
        conditions = [{"method": "equal", "value": 0, "transparency": 100},
                      {"method": "equal", "value": 1, "transparency": 50}]
        comp = MaskingCompositor("name", transparency=transparency.copy())
        assert not hasattr(comp, "transparency")
        # Transparency should be converted to conditions
        assert comp.conditions == conditions

        # conditions defined
        comp = MaskingCompositor("name", conditions=conditions.copy())
        assert comp.conditions == conditions

    def test_get_flag_value(self):
        """Test reading flag value from attributes based on a name."""
        from satpy.composites import _get_flag_value

        flag_values = da.array([1, 2])
        mask = da.array([[1, 2, 2],
                         [2, 1, 2],
                         [2, 2, 1]])
        mask = xr.DataArray(mask, dims=["y", "x"])
        flag_meanings = ["Cloud-free_land", "Cloud-free_sea"]
        mask.attrs["flag_meanings"] = flag_meanings
        mask.attrs["flag_values"] = flag_values

        assert _get_flag_value(mask, "Cloud-free_land") == 1
        assert _get_flag_value(mask, "Cloud-free_sea") == 2

        flag_meanings_str = "Cloud-free_land Cloud-free_sea"
        mask.attrs["flag_meanings"] = flag_meanings_str
        assert _get_flag_value(mask, "Cloud-free_land") == 1
        assert _get_flag_value(mask, "Cloud-free_sea") == 2

    @pytest.mark.parametrize("mode", ["LA", "RGBA"])
    def test_call_numerical_transparency_data(
            self, conditions_v1, test_data, test_ct_data, reference_data,
            reference_alpha, mode):
        """Test call the compositor with numerical transparency data.

        Use parameterisation to test different image modes.
        """
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        # Test with numerical transparency data
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v1,
                                     mode=mode)
            res = comp([test_data, test_ct_data])
        assert res.mode == mode
        for m in mode.rstrip("A"):
            np.testing.assert_allclose(res.sel(bands=m), reference_data)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    @pytest.mark.parametrize("mode", ["LA", "RGBA"])
    def test_call_numerical_transparency_data_with_3d_mask_data(
            self, test_data, value_3d_data, conditions_v3, mode):
        """Test call the compositor with numerical transparency data.

        Use parameterisation to test different image modes.
        """
        from satpy.composites import MaskingCompositor

        reference_data_v3 = test_data.where(value_3d_data[0] > 0)
        reference_alpha_v3 = xr.DataArray([[1., 0., 0.],
                                           [0., 1., 0.],
                                           [0., 0., 1.]])

        # Test with numerical transparency data using 3d test mask data which can be squeezed
        comp = MaskingCompositor("name", conditions=conditions_v3,
                                 mode=mode)
        res = comp([test_data, value_3d_data])
        assert res.mode == mode
        for m in mode.rstrip("A"):
            np.testing.assert_allclose(res.sel(bands=m), reference_data_v3)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha_v3)

    @pytest.mark.parametrize("mode", ["LA", "RGBA"])
    def test_call_numerical_transparency_data_with_3d_mask_data_exception(
            self, test_data, value_3d_data_bands, conditions_v3, mode):
        """Test call the compositor with numerical transparency data, too many dimensions to squeeze.

        Use parameterisation to test different image modes.
        """
        from satpy.composites import MaskingCompositor

        # Test with numerical transparency data using 3d test mask data which can not be squeezed
        comp = MaskingCompositor("name", conditions=conditions_v3,
                                 mode=mode)
        with pytest.raises(ValueError, match=".*Received 3 dimension\\(s\\) but expected 2.*"):
            comp([test_data, value_3d_data_bands])

    def test_call_named_fields(self, conditions_v2, test_data, test_ct_data,
                               reference_data, reference_alpha):
        """Test with named fields."""
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([test_data, test_ct_data])
        assert res.mode == "LA"
        np.testing.assert_allclose(res.sel(bands="L"), reference_data)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_call_named_fields_string(
            self, conditions_v2, test_data, test_ct_data, reference_data,
            reference_alpha):
        """Test with named fields which are as a string in the mask attributes."""
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        flag_meanings_str = "Cloud-free_land Cloud-free_sea"
        test_ct_data.attrs["flag_meanings"] = flag_meanings_str
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([test_data, test_ct_data])
        assert res.mode == "LA"
        np.testing.assert_allclose(res.sel(bands="L"), reference_data)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_method_isnan(self, test_data,
                          test_ct_data, test_ct_data_v3):
        """Test "isnan" as method."""
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        conditions_v3 = [{"method": "isnan", "transparency": 100}]

        # The data are set to NaN where ct is NaN
        reference_data_v3 = test_data.where(test_ct_data == 1)
        reference_alpha_v3 = da.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]])
        reference_alpha_v3 = xr.DataArray(reference_alpha_v3, dims=["y", "x"])
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v3)
            res = comp([test_data, test_ct_data_v3])
        assert res.mode == "LA"
        np.testing.assert_allclose(res.sel(bands="L"), reference_data_v3)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha_v3)

    def test_method_absolute_import(self, test_data, test_ct_data_v3):
        """Test "absolute_import" as method."""
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        conditions_v4 = [{"method": "absolute_import", "transparency": "satpy.resample"}]
        # This should raise AttributeError
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v4)
            with pytest.raises(AttributeError):
                comp([test_data, test_ct_data_v3])

    def test_rgb_dataset(self, conditions_v1, test_ct_data, reference_alpha):
        """Test RGB dataset."""
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        # 3D data array
        data = xr.DataArray(da.random.random((3, 3, 3)),
                            dims=["bands", "y", "x"],
                            coords={"bands": ["R", "G", "B"],
                                    "y": np.arange(3),
                                    "x": np.arange(3)})

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v1)
            res = comp([data, test_ct_data])
        assert res.mode == "RGBA"
        np.testing.assert_allclose(res.sel(bands="R"),
                                   data.sel(bands="R").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="G"),
                                   data.sel(bands="G").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="B"),
                                   data.sel(bands="B").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_rgba_dataset(self, conditions_v2, test_ct_data, reference_alpha):
        """Test RGBA dataset."""
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler
        data = xr.DataArray(da.random.random((4, 3, 3)),
                            dims=["bands", "y", "x"],
                            coords={"bands": ["R", "G", "B", "A"],
                                    "y": np.arange(3),
                                    "x": np.arange(3)})

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([data, test_ct_data])
        assert res.mode == "RGBA"
        np.testing.assert_allclose(res.sel(bands="R"),
                                   data.sel(bands="R").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="G"),
                                   data.sel(bands="G").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="B"),
                                   data.sel(bands="B").where(test_ct_data > 1))
        # The compositor should drop the original alpha band
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_incorrect_method(self, test_data, test_ct_data):
        """Test incorrect method."""
        from satpy.composites import MaskingCompositor
        conditions = [{"method": "foo", "value": 0, "transparency": 100}]
        comp = MaskingCompositor("name", conditions=conditions)
        with pytest.raises(AttributeError):
            comp([test_data, test_ct_data])
        # Test with too few projectables.
        with pytest.raises(ValueError, match="Expected 2 datasets, got 1"):
            comp([test_data])

    def test_incorrect_mode(self, conditions_v1):
        """Test initiating with unsupported mode."""
        from satpy.composites import MaskingCompositor

        # Incorrect mode raises ValueError
        with pytest.raises(ValueError, match="Invalid mode YCbCrA.  Supported modes: .*"):
            MaskingCompositor("name", conditions=conditions_v1,
                              mode="YCbCrA")


class TestNaturalEnhCompositor(unittest.TestCase):
    """Test NaturalEnh compositor."""

    def setUp(self):
        """Create channel data and set channel weights."""
        self.ch1 = xr.DataArray([1.0])
        self.ch2 = xr.DataArray([2.0])
        self.ch3 = xr.DataArray([3.0])
        self.ch16_w = 2.0
        self.ch08_w = 3.0
        self.ch06_w = 4.0

    @mock.patch("satpy.composites.NaturalEnh.__repr__")
    @mock.patch("satpy.composites.NaturalEnh.match_data_arrays")
    def test_natural_enh(self, match_data_arrays, repr_):
        """Test NaturalEnh compositor."""
        from satpy.composites import NaturalEnh
        repr_.return_value = ""
        projectables = [self.ch1, self.ch2, self.ch3]

        def temp_func(*args):
            return args[0]

        match_data_arrays.side_effect = temp_func
        comp = NaturalEnh("foo", ch16_w=self.ch16_w, ch08_w=self.ch08_w,
                          ch06_w=self.ch06_w)
        assert comp.ch16_w == self.ch16_w
        assert comp.ch08_w == self.ch08_w
        assert comp.ch06_w == self.ch06_w
        res = comp(projectables)
        assert mock.call(projectables) in match_data_arrays.mock_calls
        correct = (self.ch16_w * projectables[0] +
                   self.ch08_w * projectables[1] +
                   self.ch06_w * projectables[2])
        assert res[0] == correct
        assert res[1] == projectables[1]
        assert res[2] == projectables[2]


class TestEnhance2Dataset(unittest.TestCase):
    """Test the enhance2dataset utility."""

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_p_to_rgb(self, get_enhanced_image):
        """Test enhancing a paletted dataset in RGB mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["P"]}))
        img.palette = ((0, 0, 0), (4, 4, 4), (8, 8, 8))
        get_enhanced_image.return_value = img

        from satpy.composites import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset, convert_p=True)
        assert res.attrs["mode"] == "RGB"

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_p_to_rgba(self, get_enhanced_image):
        """Test enhancing a paletted dataset in RGBA mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["P"]}))
        img.palette = ((0, 0, 0, 255), (4, 4, 4, 255), (8, 8, 8, 255))
        get_enhanced_image.return_value = img

        from satpy.composites import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset, convert_p=True)
        assert res.attrs["mode"] == "RGBA"

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_p(self, get_enhanced_image):
        """Test enhancing a paletted dataset in P mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["P"]}))
        img.palette = ((0, 0, 0, 255), (4, 4, 4, 255), (8, 8, 8, 255))
        get_enhanced_image.return_value = img

        from satpy.composites import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset)
        assert res.attrs["mode"] == "P"
        assert res.max().values == 2

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_l(self, get_enhanced_image):
        """Test enhancing a paletted dataset in P mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["L"]}))
        get_enhanced_image.return_value = img

        from satpy.composites import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset)
        assert res.attrs["mode"] == "L"
        assert res.max().values == 1


class TestInferMode(unittest.TestCase):
    """Test the infer_mode utility."""

    def test_bands_coords_is_used(self):
        """Test that the `bands` coord is used."""
        from satpy.composites import GenericCompositor
        arr = xr.DataArray(np.ones((1, 5, 5)), dims=("bands", "x", "y"), coords={"bands": ["P"]})
        assert GenericCompositor.infer_mode(arr) == "P"

        arr = xr.DataArray(np.ones((3, 5, 5)), dims=("bands", "x", "y"), coords={"bands": ["Y", "Cb", "Cr"]})
        assert GenericCompositor.infer_mode(arr) == "YCbCr"

    def test_mode_is_used(self):
        """Test that the `mode` attribute is used."""
        from satpy.composites import GenericCompositor
        arr = xr.DataArray(np.ones((1, 5, 5)), dims=("bands", "x", "y"), attrs={"mode": "P"})
        assert GenericCompositor.infer_mode(arr) == "P"

    def test_band_size_is_used(self):
        """Test that the band size is used."""
        from satpy.composites import GenericCompositor
        arr = xr.DataArray(np.ones((2, 5, 5)), dims=("bands", "x", "y"))
        assert GenericCompositor.infer_mode(arr) == "LA"

    def test_no_bands_is_l(self):
        """Test that default (no band) is L."""
        from satpy.composites import GenericCompositor
        arr = xr.DataArray(np.ones((5, 5)), dims=("x", "y"))
        assert GenericCompositor.infer_mode(arr) == "L"


class TestLongitudeMaskingCompositor(unittest.TestCase):
    """Test case for the LongitudeMaskingCompositor compositor."""

    def test_masking(self):
        """Test longitude masking."""
        from satpy.composites import LongitudeMaskingCompositor

        area = mock.MagicMock()
        lons = np.array([-180., -100., -50., 0., 50., 100., 180.])
        area.get_lonlats = mock.MagicMock(return_value=[lons, []])
        a = xr.DataArray(np.array([1, 2, 3, 4, 5, 6, 7]),
                         attrs={"area": area, "units": "K"})

        comp = LongitudeMaskingCompositor(name="test", lon_min=-40., lon_max=120.)
        expected = xr.DataArray(np.array([np.nan, np.nan, np.nan, 4, 5, 6, np.nan]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)
        assert "units" in res.attrs
        assert res.attrs["units"] == "K"

        comp = LongitudeMaskingCompositor(name="test", lon_min=-40.)
        expected = xr.DataArray(np.array([np.nan, np.nan, np.nan, 4, 5, 6, 7]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)

        comp = LongitudeMaskingCompositor(name="test", lon_max=120.)
        expected = xr.DataArray(np.array([1, 2, 3, 4, 5, 6, np.nan]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)

        comp = LongitudeMaskingCompositor(name="test", lon_min=120., lon_max=-40.)
        expected = xr.DataArray(np.array([1, 2, 3, np.nan, np.nan, np.nan, 7]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)


def test_bad_sensor_yaml_configs(tmp_path):
    """Test composite YAML file with no sensor isn't loaded.

    But the bad YAML also shouldn't crash composite configuration loading.

    """
    from satpy.composites.config_loader import load_compositor_configs_for_sensors

    comp_dir = tmp_path / "composites"
    comp_dir.mkdir()
    comp_yaml = comp_dir / "fake_sensor.yaml"
    with satpy.config.set(config_path=[tmp_path]):
        _create_fake_composite_config(comp_yaml)

        # no sensor_name in YAML, quietly ignored
        comps, _ = load_compositor_configs_for_sensors(["fake_sensor"])
        assert "fake_sensor" in comps
        assert "fake_composite" not in comps["fake_sensor"]


def _create_fake_composite_config(yaml_filename: str):
    import yaml

    from satpy.composites import StaticImageCompositor

    with open(yaml_filename, "w") as comp_file:
        yaml.dump({
            "composites": {
                "fake_composite": {
                    "compositor": StaticImageCompositor,
                    "url": "http://example.com/image.png",
                },
            },
        },
            comp_file,
        )


class TestRealisticColors:
    """Test the SEVIRI Realistic Colors compositor."""

    def test_realistic_colors(self):
        """Test the compositor."""
        from satpy.composites import RealisticColors

        vis06 = xr.DataArray(da.arange(0, 15, dtype=np.float32).reshape(3, 5), dims=("y", "x"),
                             attrs={"foo": "foo"})
        vis08 = xr.DataArray(da.arange(15, 0, -1, dtype=np.float32).reshape(3, 5), dims=("y", "x"),
                             attrs={"bar": "bar"})
        hrv = xr.DataArray(6 * da.ones((3, 5), dtype=np.float32), dims=("y", "x"),
                           attrs={"baz": "baz"})

        expected_red = np.array([[0.0, 2.733333, 4.9333334, 6.6, 7.733333],
                                 [8.333333, 8.400001, 7.9333334, 7.0, 6.0],
                                 [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
        expected_green = np.array([
            [15.0, 12.266666, 10.066668, 8.400001, 7.2666664],
            [6.6666665, 6.6000004, 7.0666666, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0]], dtype=np.float32)

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = RealisticColors("Ni!")
            res = comp((vis06, vis08, hrv))

        arr = res.values

        assert res.dtype == np.float32
        np.testing.assert_allclose(arr[0, :, :], expected_red)
        np.testing.assert_allclose(arr[1, :, :], expected_green)
        np.testing.assert_allclose(arr[2, :, :], 3.0)


class TestFireMaskCompositor:
    """Test fire mask compositors."""

    def test_SimpleFireMaskCompositor(self):
        """Test the SimpleFireMaskCompositor class."""
        from satpy.composites import SimpleFireMaskCompositor
        rows = 2
        cols = 2
        ir_105 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                              attrs={"name": "ir_105"})
        ir_105[0, 0] = 300
        ir_38 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                             attrs={"name": "ir_38"})
        ir_38[0, 0] = 400
        nir_22 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                              attrs={"name": "nir_22"})
        nir_22[0, 0] = 100
        vis_06 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                              attrs={"name": "vis_06"})
        vis_06[0, 0] = 5

        projectables = (ir_105, ir_38, nir_22, vis_06)

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = SimpleFireMaskCompositor(
                "simple_fci_fire_mask",
                prerequisites=("ir_105", "ir_38", "nir_22", "vis_06"),
                standard_name="simple_fci_fire_mask",
                test_thresholds=[293, 20, 15, 340])
            res = comp(projectables)

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "simple_fci_fire_mask"
        assert res.data.dtype == bool

        assert np.array_equal(res.data.compute(),
                              np.array([[True, False], [False, False]]))
