import os
import sys

import argparse
import yaml
import logging

import numpy as np
from astropy.io.fits import CompImageHDU
from astropy.io import fits

import torch
from torch.utils.data.dataloader import DataLoader

from source.models.model_manager import BaseScaler
from source.dataset import FitsFileDataset
from source.data_utils import get_array_radius, get_image_from_array, plot_magnetogram

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
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--add_noise')
    parser.add_argument('--scale_factor')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--use_patches', action='store_true')
    parser.add_argument('--zero_outside', action='store_true')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--checksum', action='store_true')
    parser.add_argument('--left_right', action='store_true')

    args = parser.parse_args()

    scale_factor = 1
    if args.scale_factor:
        scale_factor = args.scale_factor

    if scale_factor != 1 or scale_factor != 2 or scale_factor != 4:
        raise RuntimeError(f'Only scale factors of 1, 2, or 4 are valid')

    instrument = args.instrument.lower()
    if instrument == 'mdi':
        if scale_factor == 1:
            run = 'checkpoints/mdi/20201019151521_HighResNet_RPRCDO_SSIMGradHistLoss_jsoc_RP_D1_18'
        if scale_factor == 2:
            run = 'checkpoints/mdi/20201018212214_HighResNet_RPRCDO_SSIMGradHistLoss_jsoc_RP_D2_18'
        if scale_factor == 4:
            run = 'checkpoints/mdi/20201020035600_HighResNet_RPRCDO_SSIMGradHistLoss_jsoc_RP_Neg_19'

    elif instrument == 'gong':
        if scale_factor == 1:
            run = 'checkpoints/gong/20201214200251_HighResNet_RPRCDO_SSIMGradHistLoss_gong_RP_19'
        if scale_factor == 2:
            run = 'checkpoints/gong/20201214200251_HighResNet_RPRCDO_SSIMGradHistLoss_gong_RP_19'
        if scale_factor == 4:
            run = 'checkpoints/gong/20201214200251_HighResNet_RPRCDO_SSIMGradHistLoss_gong_RP_19'
    else:
        raise RuntimeError(f'mdi and gong are the only valid instruments.')

    with open(run + '.yml', 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    data_config = config_data['data']
    norm = 3500
    if 'normalisation' in data_config.keys():
        norm = data_config['normalisation']

    padding = np.nan
    if args.zero_outside:
        padding = 0

    net_config = config_data['net']
    model_name = net_config['name']

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

        output_file = args.destination + '.'.join(file.replace(args.data_path,'').split('.gz')[0].split('.')[0:-1])
        os.makedirs('/'.join(output_file.split('/')[0:-1]), exist_ok=True)

        if os.path.exists(output_file + '_HR.fits') and not args.overwrite:
            logger.info(f'{file} already exists')

        else:

            file_dset = FitsFileDataset(file, 32, norm, instrument, scale_factor)

            # Try full disk
            success_sw = False
            if not args.use_patches:
                success_sw = True
                try:
                    logger.info(f'Attempting full disk inference...')

                    in_fd = np.stack([file_dset.map.data, get_array_radius(file_dset.map)], axis=0)
                    if args.left_right:
                        in_fd = np.flip(in_fd, axis=2).copy()
                    inferred = model.forward(torch.from_numpy(in_fd[None]).to(device).float()).detach().numpy()[0,...]*norm
                    if args.left_right:
                        inferred = np.fliplr(inferred).copy()
                    logger.info(f'Success.')

                except Exception as e:
                    logger.info(f'Failure. {e}')
                    success_sw = False

            if not success_sw or args.use_patches:
                logger.info(f'Attempting inference on patches...')
                dataloader = DataLoader(file_dset, batch_size=8, shuffle=False)

                output_patches = []

                for input in dataloader:

                    input = input.to(device)
                    output = model.forward(input) * norm

                    output_patches.append(output.detach().cpu().numpy())

                inferred = get_image_from_array(output_patches)
                logger.info(f'Success.')

            inferred_map = file_dset.create_new_map(inferred, scale_factor, args.add_noise, model_name, config_data, padding)

            if args.compress:
                hdu = fits.CompImageHDU(inferred_map.data, inferred_map.fits_header)
                hdu.scale(type='int32', bscale=0.1, bzero=0)
                hdu.writeto(output_file + '_HR.fits', overwrite=True, checksum=args.checksum)
            else:
                inferred_map.save(output_file + '_HR.fits', overwrite=True, checksum=args.checksum)

            if args.plot:
                plot_magnetogram(inferred_map, output_file + '_HR.png')

            del inferred_map
