import os

import torch

from google.cloud import storage
from source.utils import get_logger

logger = get_logger(__name__)

def load_from_google_cloud(run_name, epoch, model):
    """
    Construct a torch model from pe-trained model run_name stored on goolge cloud
    :param run_name: string
    :param epoch: int
    :param model: torch model
    :return: torch model with pre trained parameters
    """

    gcs_storage_client = storage.Client()

    bucket = gcs_storage_client.bucket('fdl-mag-experiments')
    blob = bucket.blob(f'checkpoints/{run_name}/epoch_{epoch}')

    if not os.path.exists(f'checkpoints/{run_name}'):
        logger.info(f'Creating checkpoint folder: checkpoints/{run_name}')
        os.makedirs(f'checkpoints/{run_name}')
    if not os.path.exists(f'checkpoints/{run_name}/epoch_{epoch}'):
        logger.info(f'Downloading checkpoint: {epoch}')
        blob.download_to_filename(f'checkpoints/{run_name}/epoch_{epoch}')

    checkpoint = torch.load(f'checkpoints/{run_name}/epoch_{epoch}', map_location='cpu')
    logger.info(f'Loading Model: fdl-mag-experiments/checkpoints/{run_name}/epoch_{epoch}')

    if list(checkpoint['model_state_dict'].keys())[0].split('.')[0] == 'module':
        state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            state_dict['.'.join(key.split('.')[1:])] = value

        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model
