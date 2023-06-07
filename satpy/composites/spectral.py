# Copyright (c) 2015-2022 Satpy developers
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
"""Composite classes for spectral adjustments."""

import logging
import warnings

import dask.array as da

from satpy.composites import CompositeBase, GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


class SpectralBlender(GenericCompositor):
    """Construct new channel by blending contributions from a set of channels.

    This class can be used to compute weighted average of different channels.
    Primarily it's used to correct the green band of AHI and FCI in order to
    allow for proper true color imagery.

    Below is an example used to generate a corrected green channel for AHI using a weighted average from
    three channels, with 63% contribution from the native green channel (B02), 29% from the red channel (B03)
    and 8% from the near-infrared channel (B04)::

      corrected_green:
        compositor: !!python/name:satpy.composites.spectral.SpectralBlender
        fractions: [0.63, 0.29, 0.08]
        prerequisites:
          - name: B02
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B03
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B04
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    Other examples can be found in the``ahi.yaml`` composite file in the satpy distribution.
    """

    def __init__(self, *args, fractions=(), **kwargs):
        """Set default keyword argument values."""
        self.fractions = fractions
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Blend channels in projectables using the weights in self.fractions."""
        if len(self.fractions) != len(projectables):
            raise ValueError("fractions and projectables must have the same length.")

        projectables = self.match_data_arrays(projectables)
        new_channel = sum(fraction * value for fraction, value in zip(self.fractions, projectables))
        new_channel.attrs = combine_metadata(*projectables)
        return super().__call__((new_channel,), **attrs)


class HybridGreen(SpectralBlender):
    """Corrector of the FCI or AHI green band.

    The green band in FCI and AHI (and other bands centered at 0.51 microns) deliberately
    misses the chlorophyll spectral reflectance local maximum at 0.55 microns
    in order to focus on aerosol and ash rather than on vegetation. This
    affects true colour RGBs, because vegetation looks brown rather than green
    and barren surface types typically gets a reddish hue.

    To correct for this the hybrid green approach proposed by Miller et al. (2016, :doi:`10.1175/BAMS-D-15-00154.2`)
    is used. The basic idea is to include some contribution also from the 0.86 micron
    channel, which is known for its sensitivity to vegetation. The formula used for this is::

      hybrid_green = (1 - F) * R(0.51) + F * R(0.86)

    where F is a constant value, that is set to 0.15 by default in Satpy.

    For example, the HybridGreen compositor can be used as follows to construct a hybrid green channel for
    AHI, with 15% contibution from the near-infrared 0.85 µm band (B04) and the remaining 85% from the native
    green 0.51 µm band (B02)::

      hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.HybridGreen
        fraction: 0.15
        prerequisites:
          - name: B02
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B04
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    Other examples can be found in the ``ahi.yaml`` and ``ami.yaml`` composite
    files in the satpy distribution.
    """

    def __init__(self, *args, fraction=0.15, **kwargs):
        """Set default keyword argument values."""
        fractions = (1 - fraction, fraction)
        super().__init__(fractions=fractions, *args, **kwargs)


class NDVIHybridGreen(SpectralBlender):
    """Construct a NDVI-weighted hybrid green channel.

    This green band correction follows the same approach as the HybridGreen compositor, but with a dynamic blend
    factor `f` that depends on the pixel-level Normalized Differece Vegetation Index (NDVI). The higher the NDVI, the
    smaller the contribution from the nir channel will be, following a liner relationship between the two ranges
    `[ndvi_min, ndvi_max]` and `limits`.

    As an example, a new green channel using e.g. FCI data and the NDVIHybridGreen compositor can be defined like::

      ndvi_hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.NDVIHybridGreen
        ndvi_min: 0.0
        ndvi_max: 1.0
        limits: [0.15, 0.05]
        prerequisites:
          - name: vis_05
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_06
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_08
            modifiers: [sunz_corrected ]
        standard_name: toa_bidirectional_reflectance

    In this example, pixels with NDVI=0.0 will be a weighted average with 15% contribution from the
    near-infrared vis_08 channel and the remaining 85% from the native green vis_05 channel, whereas
    pixels with NDVI=1.0 will be a weighted average with 5% contribution from the near-infrared
    vis_08 channel and the remaining 95% from the native green vis_05 channel. For other values of
    NDVI a linear interpolation between these values will be performed.
    """

    def __init__(self, *args, ndvi_min=0.0, ndvi_max=1.0, limits=(0.15, 0.05), **kwargs):
        """Initialize class and set the NDVI limits and the corresponding blending fraction limits."""
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.limits = limits
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Construct the hybrid green channel weighted by NDVI."""
        ndvi_input = self.match_data_arrays([projectables[1], projectables[2]])

        ndvi = (ndvi_input[1] - ndvi_input[0]) / (ndvi_input[1] + ndvi_input[0])

        ndvi.data = da.where(ndvi > self.ndvi_min, ndvi, self.ndvi_min)
        ndvi.data = da.where(ndvi < self.ndvi_max, ndvi, self.ndvi_max)

        fraction = (ndvi - self.ndvi_min) / (self.ndvi_max - self.ndvi_min) * (self.limits[1] - self.limits[0]) \
            + self.limits[0]
        self.fractions = (1 - fraction, fraction)

        return super().__call__([projectables[0], projectables[2]], **attrs)


class GreenCorrector(SpectralBlender):
    """Previous class used to blend channels for green band corrections.

    This method has been refactored to make it more generic. The replacement class is 'SpectralBlender' which computes
    a weighted average based on N number of channels and N number of corresponding weights/fractions. A new class
    called 'HybridGreen' has been created, which performs a correction of green bands centered at 0.51 microns
    following Miller et al. (2016, :doi:`10.1175/BAMS-D-15-00154.2`) in order to improve true color imagery.
    """

    def __init__(self, *args, fractions=(0.85, 0.15), **kwargs):
        """Set default keyword argument values."""
        warnings.warn(
            "'GreenCorrector' is deprecated, use 'SpectralBlender' instead, or 'HybridGreen' for hybrid green"
            " correction following Miller et al. (2016).",
            UserWarning,
            stacklevel=2
        )
        super().__init__(fractions=fractions, *args, **kwargs)


class FireMaskCompositor(CompositeBase):
    """Do some stuff."""

    @staticmethod
    def _get_closest_ecm_run(curtime):
        """ECMWF runs every 6 hours, find best match to current time."""
        pass

    @staticmethod
    def _get_ecm_time_basetime(curtime):
        """Find the closest 6-hour ECMWF forecast point.

        Forecasts are generated for 00, 06, 12, 18 UTC, but are
        only made available approximately 6 hours later.
        """
        from datetime import timedelta

        hour = curtime.hour
        rdate = curtime.replace(microsecond=0, second=0, minute=0)
        btime = 0
        if hour < 6:
            rdate = (rdate - timedelta(days=1)).strftime("%Y-%m-%d")
            btime = 18
        elif hour < 12:
            btime = 0
        elif hour < 18:
            btime = 6
        elif hour < 24:
            btime = 12

        return rdate.replace(hour=btime)

    @staticmethod
    def _get_ecm_steptimes(curtime, reftime):
        """Get step times from current time."""
        from datetime import timedelta

        tdelt = (curtime - reftime).total_seconds() / (60. * 60.)
        if tdelt <= 3:
            ptime = 0
            ntime = 3
        elif tdelt <= 6:
            ptime = 3
            ntime = 6
        elif tdelt <= 9:
            ptime = 6
            ntime = 9
        elif tdelt <= 12:
            ptime = 9
            ntime = 12
        elif tdelt <= 15:
            ptime = 12
            ntime = 15
        else:
            raise ValueError(f"Time different between ECMWF steps too large: {tdelt}!")

        # Total time between previous and next ecmwf timesteps (3 hours, in seconds)
        tot_time = 3600 * 3
        pdater = reftime + timedelta(hours=ptime)
        pastime = (curtime - pdater).total_seconds()

        n_frac = pastime / tot_time
        p_frac = 1. - n_frac

        return ptime, ntime, p_frac, n_frac

    @staticmethod
    def _read_gribs(prev_file, next_file, fracs, targ_area):
        """Read skin temperature grib files and combine."""
        # import numpy as np
        import pygrib
        import xarray as xr
        from pyresample.geometry import SwathDefinition

        # from pyresample.gradient import GradientSearchResampler
        # Load skin temp from earlier file
        grbs = pygrib.open(prev_file)
        skt_p = grbs.select(name='Skin temperature')[0].values

        # Load skin temp and lats / lons from next file (assumes lats / lons don't change).
        grbs = pygrib.open(next_file)
        grb = grbs.select(name='Skin temperature')[0]
        skt_n = grb.values
        lats, lons = grb.latlons()
        lats = xr.DataArray(lats, dims=('y', 'x'))
        lons = xr.DataArray(lons, dims=('y', 'x'))

        # Create input grid definition
        swath_def = SwathDefinition(lons=lons, lats=lats)

        # Combine skin tempsto get temperature at image time.
        skt_dt = (skt_p * fracs[0]) + (skt_n * fracs[1])

        # Resample to target area
        from pyresample import image
        swath_con = image.ImageContainerBilinear(skt_dt, swath_def, radius_of_influence=50000)
        area_con = swath_con.resample(targ_area)

        # Convert to xarray
        skt = xr.DataArray(area_con.image_data, dims=('y', 'x'))

        return skt

    def _dl_ecmwf(self, ref_time, ref_area, backup_temp):
        """Download skin temperature data from ECMWF."""
        import tempfile
        from urllib.error import HTTPError

        ecmtime = self._get_ecm_time_basetime(ref_time)
        pstep, nstep, pfrac, nfrac = self._get_ecm_steptimes(ref_time, ecmtime)

        try:
            from ecmwf.opendata import Client
            client = Client()
            temp_dir = tempfile.TemporaryDirectory()

            prev_f = f"{temp_dir.name}/previous.grib2"
            next_f = f"{temp_dir.name}/next.grib2"

            client.retrieve(date=ecmtime.strftime("%Y-%m-%d"), time=ecmtime.hour, step=pstep, type="fc",
                            param="skt", target=prev_f)
            client.retrieve(date=ecmtime.strftime("%Y-%m-%d"), time=ecmtime.hour, step=nstep, type="fc",
                            param="skt", target=next_f)
            res_skt = self._read_gribs(prev_f, next_f, (pfrac, nfrac), ref_area)

        except ImportError:
            print(f"Using default fire temperature, to use ECMWF data for fires composites requires the "
                  f"ecmwf_opendata library: pip install ecmwf_opendata"
                  f"Falling back to fixed temperature of {backup_temp}K")
        except HTTPError:
            logging.log("Warning: Fire composite requesting old data. ECMWF only holds most recent four days."
                        f"Falling back to default fire temperature of {backup_temp}K")

        return res_skt

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Generate the composite."""
        # import numpy as np
        # import scipy.ndimage

        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        skt = self._dl_ecmwf(projectables[0].attrs['start_time'], projectables[0].attrs['area'], 283)

        print(skt.shape)

        # bob = df

        # ir_105_temp = projectables[0] > 273
        # print(np.nanmin(ir_105_temp), np.nanmax(ir_105_temp))
        # temp_diff = projectables[1] - projectables[0] > 10
        # ir38_plus_nir22 = projectables[1] + projectables[2] >= 300
        # ir_38 = projectables[1] >= 310
        # proj = ir_105_temp & temp_diff & ir38_plus_nir22
        # tmp = scipy.ndimage.convolve(proj.astype(np.float32), np.ones((3, 3)), mode='constant')
        # proj = np.logical_and(tmp >= 2, proj).astype(np.float32)
        proj = skt
        proj.attrs = info
        return proj
