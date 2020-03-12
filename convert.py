import os

import torch
from torch.utils.data.dataloader import DataLoader

from source.load import load_from_google_cloud
from source.models import *
from source.dataset import FitsFileDataset

if __name__ == '__main__':
    run_name = 'to-ml-register-template_20200216025827_HighResNet_RPRC_MSELoss_64_0.0001_MSE_HRN_RP_RC_temp'
    model = HighResNet(upscale_factor=4)
    load_from_google_cloud(run_name, 16, model)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    file = '/home/mx/Documents/Xavier/FDL/Magnetograms/Data/MDI/mdi-data_mdi-prep_2011_3_13_MDI_20110313-000301.fits'
    file_dset = FitsFileDataset(file, 32)
    dataloader = DataLoader(file_dset, batch_size=8, shuffle=False)

    output_patches = []

    for input in dataloader:
        input = input.to(device)
        input =  input / 3500
        output = model.forward(input) * 3500

        output_patches.append(output.detach().cpu().numpy())


    from sunpy.cm import cm
    from source.data_utils import get_image_from_array
    import matplotlib.pyplot as plt

    out = get_image_from_array(output_patches)

    new_map = file_dset.create_new_map(out, 4)
    plt.imshow(new_map.data, cmap=cm.hmimag, vmin=-2000, vmax=2000)
    plt.show()

    os.makedirs('data', exist_ok=True)
    new_map.save('data/mdi-data_mdi-prep_2011_3_13_MDI_20110313-000301_converted.fits')

