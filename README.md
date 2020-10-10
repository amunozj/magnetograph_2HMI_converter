# Converter software to calibrate and super-resolve solar magnetograms

Currently only working on magnetograms taken by the Michelson Doppler Imager (MDI) on board the Solar and Heliospheric Observatory (SoHO) or the Global Oscillation Network Group (GONG).  Output magnetograms have the resolution and systematics of magnetograms taken by the Helioseismic and Magnetic Imager (HMI) on board the Solar Dynamics Observatory (SDO).


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3750372.svg)](https://doi.org/10.5281/zenodo.3750372)

## Installation

1. Click on the _Clone or download_ button and clone it to a repository or download it as a zip file.

![GitHub Logo](https://help.github.com/assets/images/help/repository/clone-repo-clone-url-button.png)

2. Install dependencies:

> pip install -r requirements.txt --user


## New Fits Keywords

The converter magnetogram contains the following added keywords:

***'instrume'***:  Old value is retained, but with the appended '-2HMI_HR' suffix.

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

***'--add_noise'***: Variance of the gaussian noise (in gauss) to be added to the output magnetogram.  By default no noise is added.  For noise similar to HMI use 4.7.

***'--plot'***: Make a plot of the output magnetogram.

***'--overwrite'***: Flag to overwrite files in output folder.  Files are not overwritten by default.

***'--use_patches'***:  Run inference on magnetogram patches instead of the default full disk inference.

***'--zero_outside'***: Pad outside the solar disk using zeros.  Default *np.nan*.

***'--no_rescale'***: Don't rescale the magnetogram before running inference.  By default the magnetogram is rotated and scaled to a standard plate scale that is a multiple of HMI's mean plate scale of 0.504273/pixel. 

## Example:
>python convert2HMI.py --instrument mdi --data_path /tmp/mdi/input --destination /tmp/mdi/output --use_patches --overwrite --plot
