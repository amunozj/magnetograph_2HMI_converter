import numpy as np


from torch.utils.data import Dataset
from torch import from_numpy
from sunpy.map import Map
from datetime import datetime

from source.data_utils import get_patch, get_array_radius, map_prep, scale_rotate


class FitsFileDataset(Dataset):
    """
    Construct a dataset of patches from a fits file
    """
    def __init__(self, file, size, norm, instrument, rescale, upscale_factor):

        map = map_prep(file, instrument)
        map.data[:] = map.data[:]/norm

        # Detecting need for rescale
        if rescale and np.abs(1 - (map.meta['cdelt1']/0.504273)/upscale_factor) > 0.01:
            map = scale_rotate(map, target_factor=upscale_factor)

        self.data = get_patch(map, size)
        self.map = map

    def __getitem__(self, idx):
        """
        Create torch tensor from patch with id idx
        :param idx:
        :return: tensor (W, H, 2)
        """
        patch = self.data[idx, ...]
        patch[patch != patch] = 0
        patch = from_numpy(patch).float()
        return patch

    def __len__(self):
        return self.data.shape[0]

    def create_new_map(self, new_data, scale_factor, add_noise,  model_name, config_data, padding):
        """
        Adjust header to match upscaling factor and add new keywords
        :return:
        """

        new_meta = self.map.meta.copy()

        # Changing scale and center
        new_meta['crpix1'] = new_meta['crpix1'] - self.map.data.shape[0] / 2 + self.map.data.shape[0] * scale_factor / 2
        new_meta['crpix2'] = new_meta['crpix2'] - self.map.data.shape[1] / 2 + self.map.data.shape[1] * scale_factor / 2
        new_meta['cdelt1'] = new_meta['cdelt1'] / scale_factor
        new_meta['cdelt2'] = new_meta['cdelt2'] / scale_factor

        try:
            new_meta['im_scale'] = new_meta['im_scale'] / scale_factor
            new_meta['fd_scale'] = new_meta['im_scale'] / scale_factor
            new_meta['xscale'] = new_meta['xscale'] / scale_factor
            new_meta['yscale'] = new_meta['yscale'] / scale_factor

        except:
            pass

        new_meta['naxis1'] = self.map.data.shape[0]
        new_meta['naxis2'] = self.map.data.shape[1]

        # Changing data info
        new_meta['datamin'] = np.nanmin(self.map.data)
        new_meta['datamax'] = np.nanmax(self.map.data)
        new_meta['data_rms'] = np.sqrt(np.nanmean(self.map.data**2))
        new_meta['datamean'] = np.nanmean(self.map.data)
        new_meta['datamedn'] = np.nanmedian(self.map.data)
        new_meta['dataskew'] = np.nanmedian(self.map.data)


        # Add keywords related to conversion
        try:
            new_meta['instrume'] = new_meta['instrume'] + '-2HMI_HR'
        except:
            new_meta['instrume'] = new_meta['telescop'] + '-2HMI_HR'

        new_meta['hrkey1'] = '---------------- HR ML Keywords Section ----------------'
        new_meta['date-ml'] = str(datetime.utcnow())
        new_meta['nn-model'] = model_name
        new_meta['loss'] = ', '.join('{!s}={!r}'.format(key, val) for (key, val) in config_data['loss'].items())
        new_meta['conv_doi'] = 'https://doi.org/10.5281/zenodo.3750372'
        new_meta['hrkey2'] = '---------------- HR ML Keywords Section ----------------'

        new_map = Map(new_data, new_meta)

        if add_noise:
            noise = np.random.normal(loc=0.0, scale=add_noise, size=new_map.data.shape)
            new_map.data[:] = new_map.data[:] + noise[:]

        array_radius = get_array_radius(new_map)
        new_map.data[array_radius >= 1] = padding

        return new_map




