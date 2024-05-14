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
        self.do_load_lsm = do_load_lsm
        super(FireRadiativePowerComposite, self).__init__(name, **kwargs)

    def __call__(self, projectables, **attrs):
        """Generate the composite."""
        try:
            from pyfires.PYF_basic import set_default_values, sort_l1
            from pyfires.PYF_detection import run_dets
        except ImportError:
            raise ImportError("The `pyfires` library is required to run the FireRadiativePowerComposite composite.")

        # Bugfix: The land-sea mask has `None` as an area ID. Replace this with the id from another band.
        if len(projectables) == 7:
            projectables[6].attrs["area"].area_id = projectables[5].attrs["area"].area_id
            projectables[6].attrs["area"].proj_id = projectables[5].attrs["area"].proj_id
            projectables[6].attrs["area"].description = projectables[5].attrs["area"].description

        projectables = self.match_data_arrays(projectables)

        if len(projectables) < 6:
            raise ValueError("FireRadiativePowerComposite requires input radiance data for channels near 0.6, 2.2,"
                             "3.8 and 10.5 micron. Brightness temperature data is also required for 3.8 and 10.5 "
                             "micron. If 2.2 micron isn't available, 1.6 should be used.")
        if len(projectables) == 7:
            import numpy as np
            lsm_flag = projectables[6].squeeze().astype(np.uint8)
        else:
            lsm_flag = self.do_load_lsm

        band_dict = {"vi1_band": projectables[0].attrs["name"],
                     "vi2_band": projectables[1].attrs["name"],
                     "mir_band": projectables[2].attrs["name"],
                     "lwi_band": projectables[3].attrs["name"]}

        data_dict = sort_l1(projectables[0], projectables[1],
                            projectables[2], projectables[3],
                            projectables[4], projectables[5],
                            band_dict, do_load_lsm=lsm_flag)

        data_dict = set_default_values(data_dict)

        data_dict = run_dets(data_dict, do_night=True)
        frp_out = projectables[0].copy()
        frp_out.data = data_dict["frp_est"]
        frp_out.attrs["units"] = "MW"
        frp_out.attrs["standard_name"] = "fire_radiative_power"
        frp_out.attrs["long_name"] = "Fire Radiative Power"
        frp_out.attrs["name"] = "frp"

        frp_out.attrs = combine_metadata(*projectables)

        return super().__call__((frp_out,), **attrs)
