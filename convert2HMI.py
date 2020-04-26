import argparse
import logging
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader

from source.models.model_manager import BaseScaler
from source.utils import get_logger


def get_config(instrument, fulldisk, zero_outside, add_noise, no_rescale, **kwargs):
    """
    Get config object setting values passed.

    Parameters
    ----------
    instrument : str
        Instrument name
    fulldisk : bool
        Fulldisk based inference, default is patch based
    zero_outside :
        Set region outside solar radius to zeros instead of default nan values
    add_noise : float (optional)
        Scale or standard deviation of noise to add
    no_rescale : bool (optional)
        Disable rescaling

    Returns
    -------
    tuple
        Run and config dict
    """
    if instrument == 'mdi':
        run_dir = Path('checkpoints/mdi/20200312194454_HighResNet_RPRCDO_SSIMGradHistLoss_mdi_19')
    elif instrument == 'gong':
        run_dir = Path('checkpoints/gong/20200321142757_HighResNet_RPRCDO_SSIMGradHistLoss_gong_1')

    with run_dir.with_suffix('.yml').open() as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    config_data['cli'] = {'fulldisk': fulldisk,
                          'zero_outside': zero_outside,
                          'add_noise': add_noise}

    config_data['instrument'] = instrument
    data_config = config_data['data']

    if 'normalisation' not in data_config.keys():
        data_config['normalisation'] = 3500.0

    padding = np.nan
    if zero_outside:
        padding = 0
    data_config['padding'] = padding

    rescale = True
    if no_rescale:
        rescale = False
    data_config['rescale'] = rescale

    net_config = config_data['net']

    if 'upscale_factor' not in net_config.keys():
        net_config['upscale_factor'] = 4

    return run_dir, config_data


def get_model(run, config):
    """
    Get a model based on the run and config data

    Parameters
    ----------
    run : pathlib.Path
        Path to run directory
    config : dict
        Config data

    Returns
    -------
    source.model.model_manger.TemplateModel
        The model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    amodel = BaseScaler.from_dict(config)
    amodel = amodel.to(device)
    checkpoint = torch.load(run.as_posix(), map_location='cpu')
    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module'):
            state_dict['.'.join(key.split('.')[1:])] = value
        else:
            state_dict[key] = value
    amodel.load_state_dict(state_dict)

    return amodel


def convert(in_file, out_file, config, patchsize=32):
    """
    Convert a file to HMI

    Parameters
    ----------
    in_file : pathlib.Path
        Input fits file
    out_file : pathlib.Path
        Output fits file
    config : dict
        Configuration dictionary
    patchsize : int
        Size of the patches created

    Returns
    -------

    """
    # Really slow imports so only import if we reach the point where it is needed
    from source.dataset import FitsFileDataset
    from source.data_utils import get_array_radius, get_image_from_patches, plot_magnetogram

    norm = config['data']['normalisation']
    device = config['device']
    fulldisk = config['cli']['fulldisk']

    file_dset = FitsFileDataset(in_file, patchsize, config)
    inferred = None
    # Try full disk
    if fulldisk:
        try:
            logger.info('Attempting full disk inference...')
            in_fd = np.stack([file_dset.map.data, get_array_radius(file_dset.map)], axis=0)
            inferred = model.forward(
                torch.from_numpy(in_fd[None]).to(device).float()).detach().numpy()[
                           0, ...] * norm
            logger.info('Success.')

        except Exception:
            logger.info('Full disk inference failed', exc_info=True)

    else:
        logger.info('Attempting inference on patches...')
        dataloader = DataLoader(file_dset, batch_size=8, shuffle=False)

        output_patches = []

        for patch in dataloader:
            patch.to(device)
            output = model.forward(patch) * norm

            output_patches.append(output.detach().cpu().numpy())

        inferred = get_image_from_patches(output_patches)
        logger.info(f'Success.')

    if inferred:
        inferred_map = file_dset.create_new_map(inferred, model.name)
        inferred_map.save(out_file.as_posix(), overwrite=True)

        if args.plot:
            plot_magnetogram(inferred_map, out_file.with_suffix('.png'))

        del inferred_map


if __name__ == '__main__':
    logging.root.setLevel('INFO')
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--instrument', required=True, choices=['mdi', 'gong'])
    parser.add_argument('--source_dir',  required=True, type=str)
    parser.add_argument('--destination_dir',  required=True, type=str)
    parser.add_argument('--add_noise', type=float)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--fulldisk', action='store_true')
    parser.add_argument('--zero_outside', action='store_true')
    parser.add_argument('--no_rescale', action='store_true')

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    destination_dir = Path(args.destination_dir)
    overwrite = args.overwrite

    checkpoint_dir, config_data = get_config(**vars(args))

    model = get_model(checkpoint_dir, config_data)

    source_files = chain(source_dir.rglob('*.fits'), source_dir.rglob('*.fits.gz'))

    destination_dir.mkdir(exist_ok=True, parents=True)

    for file in source_files:
        logger.info(f'Processing {file}')
        out_path = destination_dir / (file.stem + '_HR.fits')

        if out_path.exists() and not overwrite:
            logger.info(f'{file} already exists')
        else:
            convert(file, out_path, config_data)
