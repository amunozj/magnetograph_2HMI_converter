from sunpy.map import Map
from astropy.io import fits
from torch.utils.data import Dataset
from torch import from_numpy
from sunpy.map import Map

from source.data_utils import get_patch, get_array_radius


class FitsFileDataset(Dataset):
    """
    Construct a dataset of patches from a fits file
    """
    def __init__(self, file, size):
        hdul = fits.open(file, cache=False)
        hdul.verify('fix')

        if len(hdul) == 2:
            map = Map(hdul[1].data, hdul[1].header)
        else:
            map = Map(hdul[0].data, hdul[0].header)

        self.data = get_patch(map, size)
        self.map = map

    def __getitem__(self, idx):
        """
        Create torch tensor from patch with id idx
        :param idx:
        :return: tensor (W, H, 2)
        """
        patch = self.data[idx, ...]
        #patch[patch != patch] = 0
        patch = from_numpy(patch).float()
        return patch

    def __len__(self):
        return self.data.shape[0]

    def create_new_map(self, new_data, scale_factor):
        """
        Adjust header to match upscaling factor
        :return:
        """

        new_meta = self.map.meta.copy()
        new_meta['crpix1'] = new_meta['crpix1'] - self.map.data.shape[0] / 2 + self.map.data.shape[0] * scale_factor / 2
        new_meta['crpix2'] = new_meta['crpix2'] - self.map.data.shape[1] / 2 + self.map.data.shape[1] * scale_factor / 2
        new_meta['cdelt1'] = new_meta['cdelt1'] / scale_factor
        new_meta['cdelt2'] = new_meta['cdelt2'] / scale_factor

        new_map = Map(new_data, new_meta)
        array_radius = get_array_radius(new_map)

        new_map.data[array_radius >= 1] = 0

        return new_map




