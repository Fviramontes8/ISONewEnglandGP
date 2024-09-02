import numpy as np
import pandas as pd

from . import data_downloader


def read_xls(xls_filename: str, sheet: str) -> pd.DataFrame:
    '''
    Reads xls file from disk and returns a pandas dataframe

    Parameters
    ----------
    xls_filename: str
        Path to xls file to convert to a dataframe

    sheet: str, optional
        Sheetname within xls file to read and return as a dataframe
    '''
    return pd.read_excel(xls_filename, sheet_name=sheet)


def read_2011_smd_hourly_data(sheetname: str, download: bool = False):
    if download is True:
        data_downloader.download_file(
            'https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2011_smd_hourly.xls',
            'downloaded_data/2011_smd_hourly.xls',
        )
    return read_xls('downloaded_data/2011_smd_hourly.xls', sheetname)


def read_2012_smd_hourly_data(sheetname: str, download: bool = False):
    if download is True:
        data_downloader.download_file(
            'https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2012_smd_hourly.xls',
            'downloaded_data/2012_smd_hourly.xls',
        )
    return read_xls('downloaded_data/2012_smd_hourly.xls', sheetname)


def read_2013_smd_hourly_data(sheetname: str, download: bool = False):
    if download is True:
        data_downloader.download_file(
            'https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2013_smd_hourly.xls',
            'downloaded_data/2013_smd_hourly.xls',
        )
    return read_xls('downloaded_data/2013_smd_hourly.xls', sheetname)


def read_2014_smd_hourly_data(sheetname: str, download: bool = False):
    if download is True:
        data_downloader.download_file(
            'https://www.iso-ne.com/static-assets/documents/2015/05/2014_smd_hourly.xls',
            'downloaded_data/2014_smd_hourly.xls',
        )
    return read_xls('downloaded_data/2014_smd_hourly.xls', sheetname)


def read_2015_smd_hourly_data(sheetname: str, download: bool = False):
    if download is True:
        data_downloader.download_file(
            'https://www.iso-ne.com/static-assets/documents/2015/02/smd_hourly.xls',
            'downloaded_data/2015_smd_hourly.xls',
        )
    return read_xls('downloaded_data/2015_smd_hourly.xls', sheetname)


def read_2016_smd_hourly_data(sheetname: str, download: bool = False):
    if download is True:
        data_downloader.download_file(
            'https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls',
            'downloaded_data/2016_smd_hourly.xls',
        )
    return read_xls('downloaded_data/2016_smd_hourly.xls', sheetname)


def xls_to_npy_example(year: int = 2016) -> None:
    """
    Example on using the function read_xls, will read one of the ISO New
     England xls files and look for the ISONE CA sheet and extract data from
     the DEMAND column. The column data will be converted to a numpy array and
     stored as a .npy file

    Parameters
    ----------
    year: int
        Selected year to download demand data ISO New England's website
        Range of years should be 2011 - 2016
    """
    NE_data = read_xls(f'xls_data/{year}_smd_hourly.xls', 'ISONE CA')
    demand_numpy = np.array(NE_data['DEMAND'])
    np.save(f'ISONE_CA_DEMAND_{year}.npy', demand_numpy)
