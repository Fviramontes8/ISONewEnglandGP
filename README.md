# ISONewEnglandGP

Code in this repository is intended to analyze ISO New England energy, load, and demand reports. [You can find the source for getting the reports here.](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info)

More specifically, the following file is being analyzed:

  - 2011 SMD Hourly Data

## Dependencies 

### GPU version

A conda environment can be created with the following conda command:

    conda env create -f conda_envs/gpu_environment.yml

If you want to update your current environment with these dependencies run the following conda command:

    conda env update --name CUSTOM_ENV_NAME --file conda_envs/gpu_environment.yml --prune

Where `CUSTOM_ENV_NAME` is *YOUR* environment name, make sure to type `conda activate CUSTOM_ENV_NAME` for the updates to take effect.

### CPU version

    conda env create -f conda_envs/cpu_environment.yml

If you want to update your current environment with these dependencies run the following conda command:

    conda env update --name CUSTOM_ENV_NAME --file conda_envs/gpu_environment.yml --prune

Where `CUSTOM_ENV_NAME` is *YOUR* environment name, make sure to type `conda activate CUSTOM_ENV_NAME` for the updates to take effect.
