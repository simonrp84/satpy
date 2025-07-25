#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2019 Satpy developers
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

"""Testing of helper functions."""

import datetime as dt
import os
import unittest
from unittest import mock

import dask.array as da
import numpy as np
import numpy.testing
import pyresample.geometry
import pytest
import xarray as xr
from fsspec.implementations.memory import MemoryFile, MemoryFileSystem
from pyproj import CRS

from satpy.readers.core import utils as hf


class TestHelpers(unittest.TestCase):
    """Test the area helpers."""

    def test_lonlat_from_geos(self):
        """Get lonlats from geos."""
        import pyproj
        geos_area = mock.MagicMock()
        lon_0 = 0
        h = 35785831.00
        geos_area.crs = CRS({
            "a": 6378169.00,
            "b": 6356583.80,
            "h": h,
            "lon_0": lon_0,
            "proj": "geos"})

        proj = pyproj.Proj(geos_area.crs)
        expected = proj(0, 0, inverse=True)
        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(0, 0, geos_area))

        expected = proj(0, 1000000, inverse=True)
        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(0, 1000000 / h,
                                                              geos_area))

        expected = proj(1000000, 0, inverse=True)
        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(1000000 / h, 0,
                                                              geos_area))

        expected = proj(2000000, -2000000, inverse=True)
        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(2000000 / h,
                                                              -2000000 / h,
                                                              geos_area))

    def test_get_geostationary_bbox(self):
        """Get the geostationary bbox."""
        geos_area = mock.MagicMock()
        lon_0 = 0
        geos_area.crs = CRS({
            "proj": "geos",
            "lon_0": lon_0,
            "a": 6378169.00,
            "b": 6356583.80,
            "h": 35785831.00,
            "units": "m"})
        geos_area.area_extent = [-5500000., -5500000., 5500000., 5500000.]

        lon, lat = hf.get_geostationary_bounding_box(geos_area, 20)
        elon = np.array([-74.802824, -73.667708, -69.879687, -60.758081,
                         -32.224989, 32.224989, 60.758081, 69.879687,
                         73.667708, 74.802824, 74.802824, 73.667708,
                         69.879687, 60.758081, 32.224989, -32.224989,
                         -60.758081, -69.879687, -73.667708, -74.802824])

        elat = -np.array([-6.81982903e-15, -1.93889346e+01, -3.84764764e+01,
                          -5.67707359e+01, -7.18862588e+01, -7.18862588e+01,
                          -5.67707359e+01, -3.84764764e+01, -1.93889346e+01,
                          0.00000000e+00, 6.81982903e-15, 1.93889346e+01,
                          3.84764764e+01, 5.67707359e+01, 7.18862588e+01,
                          7.18862588e+01, 5.67707359e+01, 3.84764764e+01,
                          1.93889346e+01, -0.00000000e+00])

        np.testing.assert_allclose(lon, elon + lon_0)
        np.testing.assert_allclose(lat, elat)

    def test_get_geostationary_angle_extent(self):
        """Get max geostationary angles."""
        geos_area = mock.MagicMock()
        proj_dict = {
            "proj": "geos",
            "sweep": "x",
            "lon_0": -89.5,
            "a": 6378169.00,
            "b": 6356583.80,
            "h": 35785831.00,
            "units": "m"}
        geos_area.crs = CRS(proj_dict)
        expected = (0.15185342867090912, 0.15133555510297725)
        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

        proj_dict["a"] = 1000.0
        proj_dict["b"] = 1000.0
        proj_dict["h"] = np.sqrt(2) * 1000.0 - 1000.0
        geos_area.reset_mock()
        geos_area.crs = CRS(proj_dict)
        expected = (np.deg2rad(45), np.deg2rad(45))
        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

        proj_dict = {
            "proj": "geos",
            "sweep": "x",
            "lon_0": -89.5,
            "ellps": "GRS80",
            "h": 35785831.00,
            "units": "m"}
        geos_area.crs = CRS(proj_dict)
        expected = (0.15185277703584374, 0.15133971368991794)
        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

    def test_geostationary_mask(self):
        """Test geostationary mask."""
        # Compute mask of a very elliptical earth
        area = pyresample.geometry.AreaDefinition(
            "FLDK",
            "Full Disk",
            "geos",
            {"a": "6378169.0",
             "b": "3000000.0",
             "h": "35785831.0",
             "lon_0": "145.0",
             "proj": "geos",
             "units": "m"},
            101,
            101,
            (-6498000.088960204, -6498000.088960204,
             6502000.089024927, 6502000.089024927))

        mask = hf.get_geostationary_mask(area).astype(int).compute()

        # Check results along a couple of lines
        # a) Horizontal
        assert np.all(mask[50, :8] == 0)
        assert np.all(mask[50, 8:93] == 1)
        assert np.all(mask[50, 93:] == 0)

        # b) Vertical
        assert np.all(mask[:31, 50] == 0)
        assert np.all(mask[31:70, 50] == 1)
        assert np.all(mask[70:, 50] == 0)

        # c) Top left to bottom right
        assert np.all(mask[range(33), range(33)] == 0)
        assert np.all(mask[range(33, 68), range(33, 68)] == 1)
        assert np.all(mask[range(68, 101), range(68, 101)] == 0)

        # d) Bottom left to top right
        assert np.all(mask[range(101 - 1, 68 - 1, -1), range(33)] == 0)
        assert np.all(mask[range(68 - 1, 33 - 1, -1), range(33, 68)] == 1)
        assert np.all(mask[range(33 - 1, -1, -1), range(68, 101)] == 0)

    @mock.patch("satpy.readers.core.utils.AreaDefinition")
    def test_sub_area(self, adef):
        """Sub area slicing."""
        area = mock.MagicMock()
        area.pixel_size_x = 1.5
        area.pixel_size_y = 1.5
        area.pixel_upper_left = (0, 0)
        area.area_id = "fakeid"
        area.name = "fake name"
        area.proj_id = "fakeproj"
        area.crs = "some_crs"

        hf.get_sub_area(area, slice(1, 4), slice(0, 3))
        adef.assert_called_once_with("fakeid", "fake name", "fakeproj",
                                     "some_crs",
                                     3, 3,
                                     (0.75, -3.75, 5.25, 0.75))

    def test_np2str(self):
        """Test the np2str function."""
        # byte object
        npbytes = np.bytes_("hej")
        assert hf.np2str(npbytes) == "hej"

        # single element numpy array
        np_arr = np.array([npbytes])
        assert hf.np2str(np_arr) == "hej"

        # scalar numpy array
        np_arr = np.array(npbytes)
        assert hf.np2str(np_arr) == "hej"

        # multi-element array
        npbytes = np.array([npbytes, npbytes])
        with pytest.raises(ValueError, match="Array is not a string type or is larger than 1"):
            hf.np2str(npbytes)

        # non-array
        with pytest.raises(ValueError, match="Array is not a string type or is larger than 1"):
            hf.np2str(5)

    def test_get_earth_radius(self):
        """Test earth radius computation."""
        a = 2.
        b = 1.

        def re(lat):
            """Compute ellipsoid radius at the given geodetic latitude.

            Reference: Capderou, M.: Handbook of Satellite Orbits, Equation (2.20).
            """
            lat = np.deg2rad(lat)
            e2 = 1 - b ** 2 / a ** 2
            n = a / np.sqrt(1 - e2*np.sin(lat)**2)
            return n * np.sqrt((1 - e2)**2 * np.sin(lat)**2 + np.cos(lat)**2)

        for lon in (0, 180, 270):
            assert hf.get_earth_radius(lon=lon, lat=0.0, a=a, b=b) == a
        for lat in (90, -90):
            assert hf.get_earth_radius(lon=0.0, lat=lat, a=a, b=b) == b
        assert np.isclose(hf.get_earth_radius(lon=123, lat=45.0, a=a, b=b), re(45.0))

    def test_reduce_mda(self):
        """Test metadata size reduction."""
        mda = {"a": 1,
               "b": np.array([1, 2, 3]),
               "c": np.array([1, 2, 3, 4]),
               "d": {"a": 1,
                     "b": np.array([1, 2, 3]),
                     "c": np.array([1, 2, 3, 4]),
                     "d": {"a": 1,
                           "b": np.array([1, 2, 3]),
                           "c": np.array([1, 2, 3, 4])}}}
        exp = {"a": 1,
               "b": np.array([1, 2, 3]),
               "d": {"a": 1,
                     "b": np.array([1, 2, 3]),
                     "d": {"a": 1,
                           "b": np.array([1, 2, 3])}}}
        numpy.testing.assert_equal(hf.reduce_mda(mda, max_size=3), exp)

        # Make sure, reduce_mda() doesn't modify the original dictionary
        assert "c" in mda
        assert "c" in mda["d"]
        assert "c" in mda["d"]["d"]

    @mock.patch("satpy.readers.core.utils.bz2.BZ2File")
    @mock.patch("satpy.readers.core.utils.Popen")
    def test_unzip_file(self, mock_popen, mock_bz2):
        """Test the bz2 file unzipping techniques."""
        process_mock = mock.Mock()
        attrs = {"communicate.return_value": (b"output", b"error"),
                 "returncode": 0}
        process_mock.configure_mock(**attrs)
        mock_popen.return_value = process_mock

        bz2_mock = mock.MagicMock()
        bz2_mock.__enter__.return_value.read.return_value = b"TEST"
        mock_bz2.return_value = bz2_mock

        filename = "tester.DAT.bz2"
        whichstr = "satpy.readers.core.utils.which"
        segment = 3
        segmentstr = str(segment).zfill(2)
        # no pbzip2 installed with prefix
        with mock.patch(whichstr) as whichmock:
            whichmock.return_value = None
            new_fname = hf.unzip_file(filename, prefix=segmentstr)
            assert bz2_mock.__enter__.return_value.read.called
            assert os.path.exists(new_fname)
            assert os.path.split(new_fname)[1][0:2] == segmentstr
            if os.path.exists(new_fname):
                os.remove(new_fname)
        # pbzip2 installed without prefix
        with mock.patch(whichstr) as whichmock:
            whichmock.return_value = "/usr/bin/pbzip2"
            new_fname = hf.unzip_file(filename)
            assert mock_popen.called
            assert os.path.exists(new_fname)
            assert os.path.split(new_fname)[1][0:2] != segmentstr
            if os.path.exists(new_fname):
                os.remove(new_fname)

        filename = "tester.DAT"
        new_fname = hf.unzip_file(filename)
        assert new_fname is None

    @mock.patch("bz2.BZ2File")
    def test_generic_open_BZ2File(self, bz2_mock):
        """Test the generic_open method with bz2 filename input."""
        mock_bz2_open = mock.MagicMock()
        mock_bz2_open.read.return_value = b"TEST"
        bz2_mock.return_value = mock_bz2_open

        filename = "tester.DAT.bz2"
        with hf.generic_open(filename) as file_object:
            data = file_object.read()
            assert data == b"TEST"

        assert mock_bz2_open.read.called

    def test_generic_open_FSFile_MemoryFileSystem(self):
        """Test the generic_open method with FSFile in MemoryFileSystem."""
        from satpy.readers.core.remote import FSFile

        mem_fs = MemoryFileSystem()
        mem_file = MemoryFile(fs=mem_fs, path="{}test.DAT".format(mem_fs.root_marker), data=b"TEST")
        mem_file.commit()
        fsf = FSFile(mem_file)
        with hf.generic_open(fsf) as file_object:
            data = file_object.read()
            assert data == b"TEST"

    @mock.patch("satpy.readers.core.utils.open")
    def test_generic_open_filename(self, open_mock):
        """Test the generic_open method with filename (str)."""
        mock_fn_open = mock.MagicMock()
        mock_fn_open.read.return_value = b"TEST"
        open_mock.return_value = mock_fn_open

        filename = "test.DAT"
        with hf.generic_open(filename) as file_object:
            data = file_object.read()
            assert data == b"TEST"

        assert mock_fn_open.read.called

    @mock.patch("bz2.decompress", return_value=b"TEST_DECOMPRESSED")
    def test_unzip_FSFile(self, bz2_mock):
        """Test the FSFile bz2 file unzipping techniques."""
        from satpy.readers.core.remote import FSFile

        mock_bz2_decompress = mock.MagicMock()
        mock_bz2_decompress.return_value = b"TEST_DECOMPRESSED"

        segment = 3
        segmentstr = str(segment).zfill(2)

        # test zipped FSFile unzipped in fly (decompress shouldn't be called)
        mem_fs = MemoryFileSystem()
        mem_file = MemoryFile(fs=mem_fs, path="{}test.DAT.bz2".format(mem_fs.root_marker), data=b"TEST")
        mem_file.commit()
        fsf = FSFile(mem_file)

        new_fname = hf.unzip_file(fsf, prefix=segmentstr)
        mock_bz2_decompress.assert_not_called
        bz2_mock.assert_not_called
        assert os.path.exists(new_fname)
        assert os.path.split(new_fname)[1][0:2] == segmentstr
        if os.path.exists(new_fname):
            os.remove(new_fname)

        # test FSFile without unzipping in fly (decompress should be called)
        mem_file = MemoryFile(fs=mem_fs, path="{}test.DAT.bz2".format(mem_fs.root_marker),
                              data=bytes.fromhex("425A68")+b"TEST")
        mem_file.commit()
        fsf = FSFile(mem_file)

        new_fname = hf.unzip_file(fsf, prefix=segmentstr)
        mock_bz2_decompress.assert_called
        bz2_mock.assert_called
        assert os.path.exists(new_fname)
        assert os.path.split(new_fname)[1][0:2] == segmentstr
        if os.path.exists(new_fname):
            os.remove(new_fname)

    @mock.patch("os.remove")
    @mock.patch("satpy.readers.core.utils.unzip_file", return_value="dummy.txt")
    def test_pro_reading_gets_unzipped_file(self, fake_unzip_file, fake_remove):
        """Test the bz2 file unzipping context manager."""
        filename = "dummy.txt.bz2"
        expected_filename = filename[:-4]

        with hf.unzip_context(filename) as new_filename:
            assert new_filename == expected_filename

        fake_unzip_file.assert_called_with(filename)
        fake_remove.assert_called_with(expected_filename)

    def test_apply_rad_correction(self):
        """Test radiance correction technique using user-supplied coefs."""
        slope = 0.5
        offset = -0.1
        res = hf.apply_rad_correction(1.0, slope, offset)
        np.testing.assert_allclose(2.2, res)

    def test_get_user_calibration_factors(self):
        """Test the retrieval of user-supplied calibration factors."""
        radcor_dict = {"WV063": {"slope": 1.015,
                                 "offset": -0.0556},
                       "IR108": {"slo": 1.015,
                                 "off": -0.0556}}
        # Test that correct values are returned from the dict
        slope, offset = hf.get_user_calibration_factors("WV063", radcor_dict)
        assert slope == 1.015
        assert offset == -0.0556

        # Test that channels not present in dict return 1.0, 0.0
        with pytest.warns(UserWarning, match=".*not supplied coefficients.*"):
            slope, offset = hf.get_user_calibration_factors("IR097", radcor_dict)
        assert slope == 1.0
        assert offset == 0.0

        # Check that incorrect dict keys throw an error
        with pytest.raises(KeyError):
            hf.get_user_calibration_factors("IR108", radcor_dict)


class TestSunEarthDistanceCorrection:
    """Tests for applying Sun-Earth distance correction to reflectance."""

    def setup_method(self):
        """Create input / output arrays for the tests."""
        self.test_date = dt.datetime(2020, 8, 15, 13, 0, 40)

        raw_refl = xr.DataArray(da.from_array([10., 20., 40., 1., 98., 50.]),
                                attrs={"start_time": self.test_date,
                                       "scheduled_time": self.test_date})

        corr_refl = xr.DataArray(da.from_array([
            10.25484833, 20.50969667,
            41.01939333, 1.02548483,
            100.49751367, 51.27424167]),
            attrs={"start_time": self.test_date,
                   "scheduled_time": self.test_date},
        )
        self.raw_refl = raw_refl
        self.corr_refl = corr_refl

    def test_get_utc_time(self):
        """Test the retrieval of scene time from a dataset."""
        # First check correct time is returned with 'start_time'
        tmp_array = self.raw_refl.copy()
        del tmp_array.attrs["scheduled_time"]
        utc_time = hf.get_array_date(tmp_array, None)
        assert utc_time == self.test_date

        # Now check correct time is returned with 'scheduled_time'
        tmp_array = self.raw_refl.copy()
        del tmp_array.attrs["start_time"]
        utc_time = hf.get_array_date(tmp_array, None)
        assert utc_time == self.test_date

        # Now check correct time is returned with utc_date passed
        tmp_array = self.raw_refl.copy()
        new_test_date = dt.datetime(2019, 2, 1, 15, 2, 12)
        utc_time = hf.get_array_date(tmp_array, new_test_date)
        assert utc_time == new_test_date

        # Finally, ensure error is raised if no datetime is available
        tmp_array = self.raw_refl.copy()
        del tmp_array.attrs["scheduled_time"]
        del tmp_array.attrs["start_time"]
        with pytest.raises(KeyError):
            hf.get_array_date(tmp_array, None)

    def test_apply_sunearth_corr(self):
        """Test the correction of reflectances with sun-earth distance."""
        out_refl = hf.apply_earthsun_distance_correction(self.raw_refl)
        np.testing.assert_allclose(out_refl, self.corr_refl)
        assert out_refl.attrs["sun_earth_distance_correction_applied"]
        assert isinstance(out_refl.data, da.Array)

    def test_remove_sunearth_corr(self):
        """Test the removal of the sun-earth distance correction."""
        out_refl = hf.remove_earthsun_distance_correction(self.corr_refl)
        np.testing.assert_allclose(out_refl, self.raw_refl)
        assert not out_refl.attrs["sun_earth_distance_correction_applied"]
        assert isinstance(out_refl.data, da.Array)


@pytest.mark.parametrize(("data", "filename", "mode"),
                         [(b"Hello", "dummy.dat", "b"),
                          ("Hello", "dummy.txt", "t")])
def test_generic_open_binary(tmp_path, data, filename, mode):
    """Test the bz2 file unzipping context manager using dummy binary data."""
    dummy_data = data
    dummy_filename = os.fspath(tmp_path / filename)
    with open(dummy_filename, "w" + mode) as f:
        f.write(dummy_data)

    with hf.generic_open(dummy_filename, "r" + mode) as f:
        read_binary_data = f.read()

    assert read_binary_data == dummy_data

    dummy_filename = os.fspath(tmp_path / (filename + ".bz2"))
    with hf.bz2.open(dummy_filename, "w" + mode) as f:
        f.write(dummy_data)

    with hf.generic_open(dummy_filename, "r" + mode) as f:
        read_binary_data = f.read()

    assert read_binary_data == dummy_data


class TestCalibrationCoefficientPicker:
    """Unit tests for calibration coefficient selection."""

    @pytest.fixture(name="coefs")
    def fixture_coefs(self):
        """Get fake coefficients."""
        return {
            "nominal": {
                "ch1": 1.0,
                "ch2": 2.0,
            },
            "mode1": {
                "ch1": 1.1,
            },
            "mode2": {
                "ch2": 2.2,
            }
        }

    @pytest.mark.parametrize(
        ("wishlist", "expected"),
        [
            (
                None,
                {
                    "ch1": {"coefs": 1.0, "mode": "nominal"},
                    "ch2": {"coefs": 2.0, "mode": "nominal"}
                }
            ),
            (
                    "nominal",
                    {
                        "ch1": {"coefs": 1.0, "mode": "nominal"},
                        "ch2": {"coefs": 2.0, "mode": "nominal"}
                    }
            ),
            (
                {("ch1", "ch2"): "nominal"},
                {
                    "ch1": {"coefs": 1.0, "mode": "nominal"},
                    "ch2": {"coefs": 2.0, "mode": "nominal"}
                }
            ),
            (
                {"ch1": "mode1"},
                {
                    "ch1": {"coefs": 1.1, "mode": "mode1"},
                    "ch2": {"coefs": 2.0, "mode": "nominal"}
                }
            ),
            (
                {"ch1": "mode1", "ch2": "mode2"},
                {
                    "ch1": {"coefs": 1.1, "mode": "mode1"},
                    "ch2": {"coefs": 2.2, "mode": "mode2"}
                }
            ),
            (
                {"ch1": "mode1", "ch2": {"gain": 1}},
                {
                    "ch1": {"coefs": 1.1, "mode": "mode1"},
                    "ch2": {"coefs": {"gain": 1}, "mode": "external"}
                }
            ),
        ]
    )
    def test_get_coefs(self, coefs, wishlist, expected):
        """Test getting calibration coefficients."""
        picker = hf.CalibrationCoefficientPicker(coefs, wishlist)
        coefs = {
            channel: picker.get_coefs(channel)
            for channel in ["ch1", "ch2"]
        }
        assert coefs == expected

    @pytest.mark.parametrize(
        "wishlist", ["foo", {"ch1": "foo"}, {("ch1", "ch2"): "foo"}]
    )
    def test_unknown_mode(self, coefs, wishlist):
        """Test handling of unknown calibration mode."""
        with pytest.raises(KeyError, match="Unknown calibration mode"):
            hf.CalibrationCoefficientPicker(coefs, wishlist)

    @pytest.mark.parametrize(
        "wishlist", ["mode1", {"ch2": "mode1"}, {("ch1", "ch2"): "mode1"}]
    )
    def test_missing_coefs(self, coefs, wishlist):
        """Test that an exception is raised when coefficients are missing."""
        picker = hf.CalibrationCoefficientPicker(coefs, wishlist)
        with pytest.raises(KeyError, match="No mode1 calibration"):
            picker.get_coefs("ch2")

    @pytest.mark.parametrize(
        "wishlist", ["mode1", {"ch2": "mode1"}, {("ch1", "ch2"): "mode1"}]
    )
    def test_fallback_to_nominal(self, coefs, wishlist, caplog):
        """Test falling back to nominal coefficients."""
        picker = hf.CalibrationCoefficientPicker(coefs, wishlist,
                                                 fallback="nominal")
        expected = {"coefs": 2.0, "mode": "nominal"}
        assert picker.get_coefs("ch2") == expected
        assert "Falling back" in caplog.text

    def test_no_default_coefs(self):
        """Test initialization without default coefficients."""
        with pytest.raises(KeyError, match="Need at least"):
            hf.CalibrationCoefficientPicker({}, {})

    def test_no_fallback(self):
        """Test initialization without fallback coefficients."""
        with pytest.raises(KeyError, match="No fallback calibration"):
            hf.CalibrationCoefficientPicker({"nominal": 123}, {}, fallback="foo")

    def test_invalid_wishlist_type(self):
        """Test handling of invalid wishlist type."""
        with pytest.raises(TypeError, match="Unsupported wishlist type"):
            hf.CalibrationCoefficientPicker({"nominal": 123}, 123)


@pytest.mark.parametrize("name",
                         ["np2str",
                          "get_geostationary_angle_extent",
                          "get_geostationary_mask",
                          "get_geostationary_bounding_box",
                          "get_sub_area",
                          "unzip_file",
                          "unzip_context",
                          "generic_open",
                          "fromfile",
                          "bbox",
                          "get_earth_radius",
                          "reduce_mda",
                          "get_user_calibration_factors",
                          "apply_rad_correction",
                          "get_array_date",
                          "apply_earthsun_distance_correction",
                          "remove_earthsun_distance_correction",
                          "CalibrationCoefficientPicker",
                          ]
                         )
def test_init_import_warns(name):
    """Test that importing non-reader functions from __init__.py issue a warning."""
    from satpy.readers import utils

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        _ = getattr(utils, name)
