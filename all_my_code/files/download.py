def download_file(
    url,
    path=".",
    fname=None,
    decompress=True,
    premission=774,
    username=None,
    password=None,
    log_level=2,
    **kwargs,
):
    """
    A simple wrapper around the pooch package that makes downloading files easier

    Removes the need to set the hash of the file and the name is taken from the url.

    Parameters
    ----------
    url: str
        The url of the file to download
    path: str
        The destination to which the file will be downloaded. Must exist
        and must have write permission
    name: str | None
        By default [None], will get the file name from the url, or can be
        set to a string.
    decompress: bool [True]
        if the file name contains an extension that is a known compressed
        format, the file will automatically be decompressed and the
        decompressed files will be returned
    premission: int [774]
        The permission to set the download and all subfiles to.
        Must be three integer values for the file permissions - see chmod
        Does not accept four digit octal values.
        Note that permissions will be changed even if the files already exist.
    username: str | None
        if required for given url and protocol (e.g. FTP)
    password: str | None
        if required for given url and protocol (e.g. FTP)
    log_level: int [25]
        the level of logging to use. Set to level
            0 = hide all logging
            1 = show file names that do not exist
            2 = show progress bar of files that do not exist
            3 = show progress bar and all files
            4 = show all logging - including pooch
    **kwargs: key-value
        any standard inputs of pooch

    Returns
    -------
    str | list:
        if only a single entry is downloaded / decompressed, then a string will
        be returned, otherwise, a list will be returned

    """
    from .utils import change_file_permissions
    from pathlib import Path as posixpath
    import pooch
    import logging
    import sys

    log_level = 24 - log_level
    logger = pooch.get_logger()
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter("log-%(name)s | %(message)s")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    logger.addHandler(handler)

    if fname is None:
        fname = posixpath(url).name

    path = str(posixpath(path).expanduser().resolve())

    if decompress:
        decompressor = kwargs.get("processor", None)
        if decompressor is None:
            if ".zip" in url:
                kwargs["processor"] = pooch.processors.Unzip()
            elif ".tar" in url:
                kwargs["processor"] = pooch.processors.Untar()
            elif (".gz" in url) or (".bz2" in url) or (".xz" in url):
                kwargs["processor"] = pooch.processors.Decompress()

    downloader = kwargs.get("downloader", None)
    if downloader is None:
        downloader = pooch.downloaders.choose_downloader(url)
    if hasattr(downloader, "username") and username is not None:
        downloader.username = username
    if hasattr(downloader, "password") and password is not None:
        downloader.password = password

    # show the progress bar if the log level <= 20
    if log_level <= 22:
        downloader.progressbar = True
    kwargs["downloader"] = downloader

    props = dict(fname=fname, path=path)
    props.update(kwargs)

    fpath = posixpath(path).joinpath(fname)
    # if the file does not exist show if verbosity >= 25
    if fpath.is_file():
        logger.log(21, fname)
    else:
        logger.log(23, fname)
    # here we do the actual downloading
    flist = pooch.retrieve(url, None, **props)

    change_file_permissions(flist, premission)

    # return the string if it's the only item in the list
    if isinstance(flist, list):
        if len(flist) == 1:
            return flist[0]
    return flist


def get_flist_from_url(
    url,
    username=None,
    password=None,
):
    """If a url has a wildcard (*) value, remote files will be searched.

    Leverages off the `fsspec` package. This doesn't work for all HTTP urls.

    Parameters
    ----------
    url : [str]
        If a url has a wildcard (*) value, remote files will be
        searched for
    username : [str]
        if required for given url and protocol (e.g. FTP)
    password : [str]
        if required for given url and protocol (e.g. FTP)
    cache_path : [str]
        the path where the cached files will be stored. Has a special
        case where `{hash}` will be replaced with a hash based on
        the URL.
    use_cache : [bool]
        if there is a file with cached remote urls, then
        those values will be returned as a list

    Returns:
        list: a sorted list of urls
    """
    from urllib.parse import urlparse
    import fsspec

    parsed_url = urlparse(url)
    protocol = parsed_url.scheme
    host = parsed_url.netloc
    path = parsed_url.path

    props = {"protocol": protocol}
    if not protocol.startswith("http"):
        props.update({"host": host})
    if username is not None:
        props["username"] = username
    if password is not None:
        props["password"] = password

    fs = fsspec.filesystem(**props)
    if protocol.startswith("http"):
        path = f"{protocol}://{host}/{path}"

    try:
        flist = fs.glob(path)
    except AttributeError:
        raise FileNotFoundError(f"The given url does not exist: {url}")
    except TypeError:
        raise KeyError(
            f"The host {protocol}://{host} does not accept username/password"
        )

    if not protocol.startswith("https"):
        flist = [f"{protocol}://{host}{f}" for f in fs.glob(path)]

    return sorted(flist)


def download_url_tree(url, username=None, password=None, **kwargs):
    """
    Download a tree of files from a url.

    Parameters
    ----------
    url : str
        the url to download from
    username : str | None
        if required for given url and protocol (e.g. FTP)
    password : str | None
        if required for given url and protocol (e.g. FTP)
    **kwargs: key-value
        any standard inputs to `all_my_code.download_file`
    """
    import numpy as np

    if isinstance(url, str):
        url_list = get_flist_from_url(url, username, password)
    elif isinstance(url, (list, tuple, np.ndarray)):
        url_list = url

    try:
        flist = []
        for url in url_list:
            flist += (
                download_file(url, username=username, password=password, **kwargs),
            )
    except KeyboardInterrupt:
        print("\nDownloading interrupted by user. Returning partial results.")
        pass

    return flist
