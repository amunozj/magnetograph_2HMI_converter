import os
import sys

import argparse
import yaml
import logging

import numpy as np
from astropy.io import fits
import astropy.units as u
import sunpy.map

import torch
from source.models.model_manager import BaseScaler

def get_logger(name):
    """
    Return a logger for current module
    Returns
    -------

    logger : logger instance

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                  datefmt="%Y-%m-%d - %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logfile = logging.FileHandler('run.log', 'w')
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)

    return logger

if __name__ == '__main__':
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--instrument', required=True)
    parser.add_argument('--data_path')
    parser.add_argument('--destination')
    parser.add_argument('--add_noise')
    parser.add_argument('--overwrite', action='store_true')


    args = parser.parse_args()
    instrument = args.instrument.lower()

    if instrument == 'mdi':
        run = 'checkpoints/mdi/20200312194454_HighResNet_RPRCDO_SSIMGradHistLoss_mdi_19'
    elif instrument == 'gong':
        run = 'checkpoints/gong/20200321142757_HighResNet_RPRCDO_SSIMGradHistLoss_gong_1'
    else:
        raise RuntimeError(f'mdi and gong are the only valid instruments.')

    with open(run + '.yml', 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    data_config = config_data['data']
    norm = 3500
    if 'normalisation' in data_config.keys():
        norm = data_config['normalisation']

    model = BaseScaler.from_dict(config_data)

    device = torch.device("cpu")
    model = model.to(device)

    checkpoint = torch.load(run, map_location='cpu')
    try:

        try:
            model.load_state_dict(checkpoint['model_state_dict'])

        except:
            state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                state_dict['.'.join(key.split('.')[1:])] = value
            model.load_state_dict(state_dict)
    except:
        state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            state_dict['.'.join(np.append(['module'], key.split('.')[0:]))] = value
        model.load_state_dict(state_dict)


    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(args.data_path):
        list_of_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.fits') or file.endswith('.fits.gz')]

    os.makedirs(args.destination, exist_ok=True)

    for file in list_of_files:

        logger.info(f'Processing {file}')

        IN_fits = fits.open(file, cache=False)
        IN_fits.verify('fix')

        INmap = sunpy.map.Map(IN_fits[1].data, IN_fits[1].header)
        IN_fits.close()

        x, y = np.meshgrid(*[np.arange(v.value) for v in INmap.dimensions]) * u.pixel
        hpc_coords = INmap.pixel_to_world(x, y)
        rSun = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / INmap.rsun_obs

        # Mask data and rSun array to be on disk only
        INmap.data[rSun > 1] = 0
        rSun[rSun > 1] = 0

        # Load data into correct format and normalise
        in_fd = np.stack([INmap.data / norm, rSun], axis=0)
        in_fd = in_fd[None]

        # Transform to tensor and send to CPU
        in_fd_t = torch.from_numpy(in_fd).to(device).float()

        # Set model to eval mode and run FD inference
        scale_factor = config_data['net']['upscale_factor']

        new_meta = INmap.meta.copy()
        new_meta['crpix1'] = new_meta['crpix1'] - INmap.data.shape[0] / 2 + INmap.data.shape[0] * scale_factor / 2
        new_meta['crpix2'] = new_meta['crpix2'] - INmap.data.shape[1] / 2 + INmap.data.shape[1] * scale_factor / 2
        new_meta['cdelt1'] = new_meta['cdelt1'] / scale_factor
        new_meta['cdelt2'] = new_meta['cdelt2'] / scale_factor

        inferred_map = sunpy.map.Map(
            model.forward(in_fd_t).detach().numpy()[0, ...] * norm, new_meta)
        x, y = np.meshgrid(*[np.arange(v.value) for v in inferred_map.dimensions]) * u.pixel
        hpc_coords = inferred_map.pixel_to_world(x, y)
        rSunI = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / inferred_map.rsun_obs

        if args.add_noise:
            noise = np.random.normal(loc=0.0, scale=args.add_noise, size=inferred_map.data.shape)
            inferred_map.data[:] = inferred_map.data[:] + noise[:]

        inferred_map.data[rSunI > 1] = 0
        try:
            inferred_map.save(args.destination + '/' + file.split('/')[-1].split('.')[0] + '_HR.fits')
        except:
            logger.info(f'{file} already exists')

        del inferred_map
