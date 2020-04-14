# Converter software to calibrate and super-resolve solar magnetograms

Currently only working on magnetograms taken by the Michelson Doppler Imager (MDI) on board the Solar and Heliospheric Observatory (SoHO) or the Global Oscillation Network Group (GONG).  Output magnetograms have the resolution and systematics of magnetograms taken by the Helioseismic and Magnetic Imager (HMI) on board the Solar Dynamics Observatory (SDO).

## Installation

1. Click on the _Clone or download_ button and clone it to a repository or download it as a zip file.

![GitHub Logo](https://help.github.com/assets/images/help/repository/clone-repo-clone-url-button.png)

2. Install dependencies:

> pip install -r requirements.txt --user


## New Fits Keywords

The converter magnetogram contains the following added keywords:

***'telescop'***:  Old value is retained, but with the appended '-2HMI_HR' suffix.

***'date-conv'***:  UTC date of conversion.

***'nn-model'***:  Name of neural network architecture.

***'loss'***:  Loss parameters.

***'converter_doi'***: DOI pointing to this converter's repository

## Usage Arguments

### Required

***'--instrument'***: Input instrument.  Currently only **'gong'** and **'mdi'** are valid choices.

***'--data_path'***: Folder containing the files to be converted.

***'--destination'***: Destination of the super-resolved magnetograms.

### Optional

***'--add_noise'***: Variance of the gaussian noise (in gauss) to be added to the output magnetogram.  By default no noise is added.  For HMI noise use 4.7.

***'--overwrite'***: Flag to overwrite files in output folder.  Files are not overwritten by default.

***'--use_patches'***:  Run inference on magnetogram patches instead of the default full disk inference.

***'--zero_outside'***: Padd outside the solar disk using zeros.  Default *np.nan*.

## Example:
>python convert2HMI.py --instrument gong --data_path /tmp/gong/input --destination /tmp/gong/output --use_patches --overwrite
