from os import PathLike
from typing import Any, Optional, Union
from urllib.request import urlretrieve


def download_file(
    url: str, file_rename: Optional[Union[str, PathLike[Any], None]] = None
) -> None:
    """
    Downloads file at specified url

    Parameters
    ----------
    url: str
        Url to download a file from. The url is assumed to be non-interactive
            like the Linux utility wget
    file_rename: Optional[Union[str, PathLike[Any], None]], optional
        Filename to rename the downloaded file
    """
    if file_rename:
        _ = urlretrieve(url, filename=file_rename)
    else:
        _ = urlretrieve(url)
