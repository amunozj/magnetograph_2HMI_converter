from datetime import datetime

import numpy as np
from sunpy.map import Map
from torch import from_numpy
from torch.utils.data import Dataset

from source.data_utils import get_patches, get_array_radius, map_prep, scale_rotate


class FitsFileDataset(Dataset):
    """
    Construct a dataset of patches from a fits file
    """

    def __init__(self, file, size, config_data):

        self.instrument = config_data['instrument']
        self.norm = config_data['data']['normalisation']
        self.rescale = config_data['data']['rescale']
        self.upscale_factor = config_data['net']['upscale_factor']
        self.padding = config_data['data']['padding']
        self.loss = config_data['loss']
        self.add_noise = config_data['cli']['add_noise']

        amap = map_prep(file, self.instrument)
        amap.data[:] = amap.data[:] / self.norm

        # Detecting need for rescale
        if self.rescale and np.abs(1 - (amap.meta['cdelt1']/0.504273) / self.upscale_factor) > 0.01:
            amap = scale_rotate(amap, target_factor=self.upscale_factor)

        self.data = get_patches(amap, size)
        self.map = amap

    def __getitem__(self, idx):
        """
        Get a patch

        Parameters
        ----------
        idx : int
            Index

        Returns
        -------
        torch.tensor
            patch
        """
        patch = self.data[idx, ...]
        patch[patch != patch] = 0
        patch = from_numpy(patch).float()
        return patch

    def __len__(self):
        return self.data.shape[0]

    def create_new_map(self, new_data, model_name):
        """
        Adjust header to match upscaling factor and add new keywords

        Parameters
        ----------
        new_data :
            Superresolved map data
        model_name : str
            Name of the model use
        """

        new_meta = self.map.meta.copy()

        # Changing scale and center
        new_meta['crpix1'] = (new_meta['crpix1'] - self.map.data.shape[0] / 2
                              + self.map.data.shape[0] * self.upscale_factor / 2)
        new_meta['crpix2'] = (new_meta['crpix2'] - self.map.data.shape[1] / 2
                              + self.map.data.shape[1] * self.upscale_factor / 2)
        new_meta['cdelt1'] = new_meta['cdelt1'] / self.upscale_factor
        new_meta['cdelt2'] = new_meta['cdelt2'] / self.upscale_factor

        try:
            new_meta['im_scale'] = new_meta['im_scale'] / self.upscale_factor
            new_meta['fd_scale'] = new_meta['im_scale'] / self.upscale_factor
            new_meta['xscale'] = new_meta['xscale'] / self.upscale_factor
            new_meta['yscale'] = new_meta['yscale'] / self.upscale_factor
        except:
            pass

        new_meta['naxis1'] = self.map.data.shape[0]
        new_meta['naxis2'] = self.map.data.shape[1]

        # Changing data info
        new_meta['datamin'] = np.nanmin(self.map.data)
        new_meta['datamax'] = np.nanmax(self.map.data)
        new_meta['data_rms'] = np.sqrt(np.nanmean(self.map.data ** 2))
        new_meta['datamean'] = np.nanmean(self.map.data)
        new_meta['datamedn'] = np.nanmedian(self.map.data)
        new_meta['dataskew'] = np.nanmedian(self.map.data)

        # Add keywords related to conversion
        try:
            new_meta['instrume'] = new_meta['instrume'] + '-2HMI_HR'
        except KeyError:
            new_meta['instrume'] = new_meta['telescop'] + '-2HMI_HR'

        new_meta['hrkey1'] = '---------------- HR ML Keywords Section ----------------'
        new_meta['date-ml'] = str(datetime.utcnow())
        new_meta['nn-model'] = model_name
        new_meta['loss'] = ', '.join(
            '{!s}={!r}'.format(key, val) for (key, val) in self.loss.items())
        new_meta['conv_doi'] = 'https://doi.org/10.5281/zenodo.3750372'
        new_meta['hrkey2'] = '---------------- HR ML Keywords Section ----------------'

        new_map = Map(new_data, new_meta)

        if self.add_noise:
            noise = np.random.normal(loc=0.0, scale=self.add_noise, size=new_map.data.shape)
            new_map.data[:] = new_map.data[:] + noise[:]

        array_radius = get_array_radius(new_map)
        new_map.data[array_radius >= 1] = self.padding

        return new_map
