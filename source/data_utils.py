import numpy as np
import math

from astropy import units as u
from sklearn.feature_extraction import image
from sunpy.map import all_pixel_indices_from_map

from source.utils import disable_warnings, get_logger

disable_warnings()
logger = get_logger(__name__)


def get_patch(amap, size):
    """
    create patches of dimension size * size with a defined stride.
    Since stride is equals to size, there is no overlap

    Parameters
    ----------
    amap: sunpy map

    size: integer
        size of each patch

    Returns
    -------
    numpy array [num_patches, num_channel, size, size]
        channels are magnetic field, radial distance relative to radius

    """
    array_radius = get_array_radius(amap)

    patches = image.extract_patches(amap.data, (size, size), extraction_step=size)
    patches = patches.reshape([-1] + list((size, size)))

    patches_r = image.extract_patches(array_radius, (size, size), extraction_step=size)
    patches_r = patches_r.reshape([-1] + list((size, size)))

    return np.stack([patches, patches_r], axis=1)

def get_array_radius(amap):
    """
    Compute an array with the radial coordinate for each pixel
    :param amap:
    :return: (W, H) array
    """
    x, y = np.meshgrid(*[np.arange(v.value) for v in amap.dimensions]) * u.pixel
    hpc_coords = amap.pixel_to_world(x, y)
    array_radius = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / amap.rsun_obs

    return array_radius


def get_image_from_array(list_patches):
    """
    Reconstruct from a list of patches the full disk image
    :param list_patches:
    :return:
    """
    out = np.array(list_patches)
    out_r = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])

    size = int(math.sqrt(out_r.shape[0]))
    out_array = np.array_split(out_r, size, axis=0)
    out_array = np.concatenate(out_array, axis=1)
    out_array = np.concatenate(out_array, axis=1)

    return out_array
