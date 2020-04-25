import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader

from source.data_utils import get_array_radius, get_image_from_patches, plot_magnetogram
from source.dataset import FitsFileDataset
from source.models.model_manager import BaseScaler
from source.utils import get_logger

if __name__ == '__main__':
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--instrument', required=True, choices=['mdi', 'gong'])
    parser.add_argument('--data_path')
    parser.add_argument('--destination')
    parser.add_argument('--add_noise')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--use_patches', action='store_true')
    parser.add_argument('--zero_outside', action='store_true')
    parser.add_argument('--no_rescale', action='store_true')

    args = parser.parse_args()
    instrument = args.instrument.lower()

    if args.instrument == 'mdi':
        run = 'checkpoints/mdi/20200312194454_HighResNet_RPRCDO_SSIMGradHistLoss_mdi_19'
    elif args.instrument == 'gong':
        run = 'checkpoints/gong/20200321142757_HighResNet_RPRCDO_SSIMGradHistLoss_gong_1'

    with open(run + '.yml', 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    data_config = config_data['data']
    norm = 3500
    if 'normalisation' in data_config.keys():
        norm = data_config['normalisation']

    padding = np.nan
    if args.zero_outside:
        padding = 0

    rescale = True
    if args.no_rescale:
        rescale = False

    net_config = config_data['net']
    model_name = net_config['name']
    upscale_factor = 4
    if 'upscale_factor' in net_config.keys():
        upscale_factor = net_config['upscale_factor']

    model = BaseScaler.from_dict(config_data)

    device = torch.device("cpu")
    model = model.to(device)

    checkpoint = torch.load(run, map_location='cpu')

    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module'):
            state_dict['.'.join(key.split('.')[1:])] = value
        else:
            state_dict[key] = value

    model.load_state_dict(state_dict)

    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(args.data_path):
        list_of_files += [os.path.join(dirpath, file) for file in filenames if
                          file.endswith('.fits') or file.endswith('.fits.gz')]

    os.makedirs(args.destination, exist_ok=True)

    for file in list_of_files:

        logger.info(f'Processing {file}')

        output_file = args.destination + '/' + '.'.join(
            file.split('/')[-1].split('.gz')[0].split('.')[0:-1])
        if os.path.exists(output_file + '_HR.fits') and not args.overwrite:
            logger.info(f'{file} already exists')

        else:

            file_dset = FitsFileDataset(file, 32, norm, instrument, rescale, upscale_factor)

            # Try full disk
            success_sw = False
            if not args.use_patches:
                success_sw = True
                try:
                    logger.info(f'Attempting full disk inference...')
                    in_fd = np.stack([file_dset.map.data, get_array_radius(file_dset.map)], axis=0)
                    inferred = model.forward(
                        torch.from_numpy(in_fd[None]).to(device).float()).detach().numpy()[
                                   0, ...] * norm
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

                inferred = get_image_from_patches(output_patches)
                logger.info(f'Success.')

            inferred_map = file_dset.create_new_map(inferred, upscale_factor, args.add_noise,
                                                    model_name, config_data, padding)
            inferred_map.save(output_file + '_HR.fits', overwrite=True)

            if args.plot:
                plot_magnetogram(inferred_map, output_file + '_HR.png')

            del inferred_map
