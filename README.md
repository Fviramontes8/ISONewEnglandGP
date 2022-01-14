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

## Dependancies (installing with conda is recommended)

### Numpy
For numerical computations


    pip3 install numpy


or


    conda install numpy

### Pandas
For reading in the [source data](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info)

    pip3 install pandas

or

    conda install pandas

### Scipy
For use in butterworth filter

    pip3 install scipy

or

    conda install scipy

### Matplotlib
For plotting data

    pip3 install matplotlib

or

    conda install matplotlib

### PyTorch
For creating tensors and to use with GPyTorch

It is recommend to [go to the PyTorch website for the most up-to-date information on installing PyTorch](https://pytorch.org)
#### GPU

    pip3 install torch torchvision

or

    conda install pytorch torchvision cuda=XX.X -c pytorch

Where `XX.X` is a CUDA version (e.g. 9.2, 10.1, 10.2, 11.3)

#### CPU

    pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

or

    conda install pytorch torchvision cpuonly -c pytorch

### GPyTorch

    pip install gpytorch
