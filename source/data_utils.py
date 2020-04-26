import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
from sunpy.cm import cm as scm
from sunpy.map import Map

from source.utils import disable_warnings, get_logger

disable_warnings()
logger = get_logger(__name__)


def map_prep(file, instrument):
    """
    Return a processed magnetogram

    Parameters
    ----------
    file : str
        File to prep
    instrument : str
        Instrument name

    Returns
    -------
    sunpy.map.Map
        Preped map and filepath
    """

    # Open fits file as HUDL and fix header
    hdul = fits.open(file, cache=False)
    hdul.verify('fix')

    # Assemble Sunpy map (compressed fits file so use second hdu)

    if len(hdul) == 2:
        sun_map = Map(hdul[1].data, hdul[1].header)
        data = sun_map.data.astype('>f4')
        header = sun_map.meta
    else:
        data = hdul[0].data
        header = hdul[0].header

    if instrument == 'mdi':
        if 'SOLAR_P0' in header:
            header['RSUN_OBS'] = header['OBS_R0']
            header['RSUN_REF'] = 696000000
            header['CROTA2'] = -header['SOLAR_P0']
            header['CRVAL1'] = 0.000000
            header['CRVAL2'] = 0.000000
            header['CUNIT1'] = 'arcsec'
            header['CUNIT2'] = 'arcsec'
            header['DSUN_OBS'] = header['OBS_DIST']
            header['DSUN_REF'] = 1

        if 'DSUN_REF' not in header:
            header['DSUN_REF'] = u.au.to('m')

        try:
            header.pop('SOLAR_P0')
            header.pop('OBS_DIST')
            header.pop('OBS_R0')
        except KeyError:
            pass

    elif instrument == 'gong':
        if len(header['DATE-OBS']) < 22:
            header['RSUN_OBS'] = header['RADIUS'] * 180 / np.pi * 60 * 60
            header['RSUN_REF'] = 696000000
            header['CROTA2'] = 0
            header['CUNIT1'] = 'arcsec'
            header['CUNIT2'] = 'arcsec'
            header['DSUN_OBS'] = header['DISTANCE'] * 149597870691
            header['DSUN_REF'] = 149597870691
            header['cdelt1'] = 2.5534
            header['cdelt2'] = 2.5534

            header['CTYPE1'] = 'HPLN-TAN'
            header['CTYPE2'] = 'HPLT-TAN'

            date = header['DATE-OBS']
            header['DATE-OBS'] = date[0:4] + '-' + date[5:7] + '-' + date[8:10] + 'T' +\
                header['TIME-OBS'][0:11]

    sun_map = Map(data, header)

    return sun_map


def scale_rotate(amap, target_scale=0.504273, target_factor=0):
    """

    Parameters
    ----------
    amap : sunpy.map.Map
        Input map
    target_factor : float

    target_scale : int


    Returns
    -------
    sunpy.map.Map
        Scaled and rotated map
    """

    scalex = amap.meta['cdelt1']
    scaley = amap.meta['cdelt2']
    if scalex != scaley:
        raise ValueError('Square pixels expected')

    # Calculate target factor if not provided
    if target_factor == 0:
        target_factor = np.round(scalex / target_scale)

    ratio_plate = target_factor * target_scale / scalex
    logger.debug(np.round(scalex / target_scale) / scalex * target_scale)
    ratio_dist = amap.meta['dsun_obs'] / amap.meta['dsun_ref']
    logger.debug(ratio_dist)

    # Pad image, if necessary
    new_shape = int(4096/target_factor)

    # Reform map to new size if original shape is too small
    if new_shape > amap.data.shape[0]:

        new_fov = np.zeros((new_shape, new_shape)) * np.nan
        new_meta = amap.meta

        new_meta['crpix1'] = new_meta['crpix1'] - amap.data.shape[0] / 2 + new_fov.shape[0] / 2
        new_meta['crpix2'] = new_meta['crpix2'] - amap.data.shape[1] / 2 + new_fov.shape[1] / 2

        # Identify the indices for appending the map original FoV
        i1 = int(new_fov.shape[0] / 2 - amap.data.shape[0] / 2 + 1)
        i2 = int(new_fov.shape[0] / 2 + amap.data.shape[0] / 2 + 1)

        # Insert original image in new field of view
        new_fov[i1:i2, i1:i2] = amap.data[:, :]

        # Assemble Sunpy map
        amap = Map(new_fov, new_meta)

    # Rotate solar north up rescale and recenter
    rot_map = amap.rotate(scale=ratio_dist / ratio_plate, recenter=True)

    # Want image the same size as original so use dimension from input map
    x_scale = ((rot_map.scale.axis1 * amap.dimensions.x) / 2)
    y_scale = ((rot_map.scale.axis2 * amap.dimensions.y) / 2)

    logger.debug(f'x-scale {x_scale}, y-scale {y_scale}')

    if x_scale != y_scale:
        logger.error(f'x-scale: {x_scale} and y-scale {y_scale} do not match')

    # Define coordinates
    bottom_left = SkyCoord(-x_scale, -y_scale, frame=rot_map.coordinate_frame)
    top_right = SkyCoord(x_scale, y_scale, frame=rot_map.coordinate_frame)

    crop_map = rot_map.submap(bottom_left, top_right)

    # Update Meta
    crop_map.meta['cdelt1'] = target_factor * target_scale
    crop_map.meta['cdelt2'] = target_factor * target_scale
    crop_map.meta['rsun_obs'] = crop_map.meta['rsun_obs'] * ratio_dist
    crop_map.meta['dsun_obs'] = crop_map.meta['dsun_ref']

    crop_map.meta['im_scale'] = target_factor * target_scale
    crop_map.meta['fd_scale'] = target_factor * target_scale
    crop_map.meta['xscale'] = target_factor * target_scale
    crop_map.meta['yscale'] = target_factor * target_scale

    return crop_map


def get_patches(amap, size):
    """
    create patches of dimension size * size with a stride of size.

    Since stride is equals to size, there is no overlap

    Parameters
    ----------
    amap : sunpy.map.Map
        Input map to create tiles from

    size : int
        Size of each patch

    Returns
    -------
    numpy.array (num_patches, num_channel, size, size)
        Channels are magnetic field, radial distance relative to radius

    """
    array_radius = get_array_radius(amap)

    patches = image.extract_patches(amap.data, (size, size), extraction_step=size)
    patches = patches.reshape(-1, size, size)

    patches_r = image.extract_patches(array_radius, (size, size), extraction_step=size)
    patches_r = patches_r.reshape(-1, size, size)

    return np.stack([patches, patches_r], axis=1)


def get_array_radius(amap):
    """
    Compute an array with the radial coordinate for each pixel.

    Parameters
    ----------
    amap : sunpy.map.Map
        Input map
    Returns
    -------
    numpy.ndarray
        Array full of radius values
    """
    x, y = np.meshgrid(*[np.arange(v.value) for v in amap.dimensions]) * u.pixel
    hpc_coords = amap.pixel_to_world(x, y)
    array_radius = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / amap.rsun_obs

    return array_radius


def get_image_from_patches(list_patches):
    """
    Reconstruct from a list of patches the full disk image

    Parameters
    ----------
    list_patches : list of patches produced from `get_patches`

    Returns
    -------
    numpy.ndarray
        Reconstructed image
    """
    out = np.array(list_patches)
    out_r = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])

    size = int(np.sqrt(out_r.shape[0]))
    out_array = np.array_split(out_r, size, axis=0)
    out_array = np.concatenate(out_array, axis=1)
    out_array = np.concatenate(out_array, axis=1)

    return out_array


def plot_magnetogram(amap, file, scale=1, vmin=-2000, vmax=2000, cmap=scm.hmimag):
    """
    Plot magnetogram

    Parameters
    ----------
    amap : sunpy.map.Map
        Magnetogram map to plot
    file : pathlib.Path
        Filename to save plot as
    scale :

    vmin : float
        Min value for color map
    vmax : float
        Max value for color map
    cmap : matplotlib.colors.Colormap
        Colormap
    """

    # Size definitions
    dpi = 400
    pxx = amap.data.shape[0] * scale  # Horizontal size of each panel
    pxy = pxx  # Vertical size of each panel

    nph = 1  # Number of horizontal panels
    npv = 1  # Number of vertical panels

    # Padding
    padv = 0  # Vertical padding in pixels
    padv2 = 0  # Vertical padding in pixels between panels
    padh = 0  # Horizontal padding in pixels at the edge of the figure
    padh2 = 0  # Horizontal padding in pixels between panels

    # Figure sizes in pixels
    fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in pixels
    fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in pixels

    # Conversion to relative units
    ppxx = pxx / fszh  # Horizontal size of each panel in relative units
    ppxy = pxy / fszv  # Vertical size of each panel in relative units
    ppadv = padv / fszv  # Vertical padding in relative units
    # Never used so commented out
    # ppadv2 = padv2 / fszv  # Vertical padding in relative units
    ppadh = padh / fszh  # Horizontal padding the edge of the figure in relative units
    # Never used so commented out
    # ppadh2 = padh2 / fszh  # Horizontal padding between panels in relative units

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

    # ## Add Perihelion
    ax1 = fig.add_axes([ppadh + ppxx, ppadv + ppxy, ppxx, ppxy])
    ax1.imshow(amap.data, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax1.set_axis_off()
    ax1.text(0.99, 0.99, 'ML Output', horizontalalignment='right', verticalalignment='top',
             color='k', transform=ax1.transAxes)

    fig.savefig(file, bbox_inches='tight', dpi=dpi, pad_inches=0)
