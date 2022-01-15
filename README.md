# ISONewEnglandGP

Code in this repository is intended to analyze ISO New England energy, load, and demand reports. [You can find the source for getting the reports here.](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info)

More specifically, the following files are being analyzed:

  - 2021 SMD Hourly Data
  - 2020 SMD Hourly Data
  - 2019 SMD Hourly Data
  - 2018 SMD Hourly Data
  - 2017 SMD Hourly Data
  - 2016 SMD Hourly Data
  - 2015 SMD Hourly Data
  - 2014 SMD Hourly Data
  - 2013 SMD Hourly Data
  - 2012 SMD Hourly Data
  - 2011 SMD Hourly Data

## Dependencies 

### GPU version

A conda environment can be created with the following conda command:

    conda env create -f gpu_environment.yml

If you want to update your current environment with these dependencies run the following conda command:

    conda env update --name CUSTOM_ENV_NAME --file gpu_environment.yml --prune

Where `CUSTOM_ENV_NAME` is *YOUR* environment name

### CPU version

    conda env create -f cpu_environment.yml

If you want to update your current environment with these dependencies run the following conda command:

    conda env update --name CUSTOM_ENV_NAME --file gpu_environment.yml --prune

Where `CUSTOM_ENV_NAME` is *YOUR* environment name
