# Copyright (c) 2023 Satpy developers
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

from satpy.composites import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


class MultiFireCompositor(GenericCompositor):
    """Combines an RGB image, such as true_color, with a fire radiative power dataset."""

    def __init__(self, min_frp=0., max_frp=30., **kwargs):
        """Collect custom configuration values.

        Args:
            min_frp (float): Minimum value of FRP to use in the composite.
            max_frp (float): Maximum value of FRP to use in the composite.
            **kwargs: Additional keyword arguments passed to GenericCompositor.
        """
        self.min_frp = min_frp
        self.max_frp = max_frp

        super(MultiFireCompositor, self).__init__(**kwargs)

    def __call__(self, projectables, **kwargs):
        """Generate the composite."""
        import numpy as np

        projectables = self.match_data_arrays(projectables)
        # At least one composite is requested.

        the_frp = projectables[0]

        frp_data = the_frp.data

        frp_data = np.where(np.isfinite(frp_data), frp_data, np.nan)
        frp_data = np.where(frp_data > self.max_frp, self.max_frp, frp_data)
        frp_data = np.where(frp_data < self.min_frp, self.min_frp, frp_data)
        frp_data = np.where(frp_data < self.min_frp, np.nan, frp_data)
        frp_data = frp_data / self.max_frp


        frp_data = frp_data / np.nanmax(frp_data)

        the_frp.data = frp_data * 255.

        return super(MultiFireCompositor, self).__call__([the_frp], **kwargs)


class FireRadiativePowerComposite(GenericCompositor):
    """Combines an RGB image, such as true_color, with a fire radiative power dataset."""

    def __init__(self, name, do_load_lsm=True, **kwargs):
        """Collect custom configuration values."""
        self.do_load_lsm = False#do_load_lsm
        super(FireRadiativePowerComposite, self).__init__(name, **kwargs)


    def __call__(self, projectables, **attrs):
        """Generate the composite."""
        from pyfires.PYF_basic import sort_l1
        from pyfires.PYF_detection import run_dets

        projectables = self.match_data_arrays(projectables)
        # At least one composite is requested.

        if len(projectables) != 5:
            raise ValueError("FireRadiativePowerComposite requires input radiance data for channels near 0.6, "
                             "3.8 and 10.5 micron. Brightness temperature data is also required for 3.8 and 10.5 "
                             "micron.")

        band_dict = {"vi1_band": projectables[0].attrs["name"],
                     "mir_band": projectables[1].attrs["name"],
                     "lwi_band": projectables[2].attrs["name"]}

        data_dict = sort_l1(projectables[0], projectables[1],
                            projectables[2], projectables[3],
                            projectables[4],
                            band_dict, do_load_lsm=self.do_load_lsm)


        fire_dets, frp_data = run_dets(data_dict)

        frp_out = projectables[0].copy()
        frp_out.data = frp_data
        frp_out.attrs["units"] = "MW"
        frp_out.attrs["standard_name"] = "fire_radiative_power"
        frp_out.attrs["long_name"] = "Fire Radiative Power"
        frp_out.attrs["name"] = "frp"


        frp_out.attrs = combine_metadata(*projectables)

        return super().__call__((frp_out,), **attrs)
