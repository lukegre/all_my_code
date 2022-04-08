import posix


def download_file(url, path='.', fname=None, progress=True, decompress=True, premission=774, username=None, password=None, **kwargs):
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
    progress: bool [True]
        Show a progress bar for downloading without having to specify the 
        downloader. 
    decompress: bool [True]
        if the file name contains an extension that is a known compressed 
        format, the file will automatically be decompressed and the 
        decompressed files will be returned 
    premission: int [774]
        The permission to set the download and all subfiles to. 
        Must be three integer values for the file permissions - see chmod
        Does not accept four digit octal values. 
        Note that permissions will be changed even if the files already exist. 
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
    import os
    
    if fname is None:
        fname = posixpath(url).name
        
    if progress:
        downloader = kwargs.get('downloader', None)
        if downloader is None:
            downloader = pooch.downloaders.choose_downloader(url)
        downloader.progressbar = True
        if hasattr(downloader, 'username') and username is not None:
            downloader.username = username
        if hasattr(downloader, 'password') and password is not None:
            downloader.password = password
            print(downloader)
        kwargs['downloader'] = downloader
    
    if decompress:
        decompressor = kwargs.get('processor', None)
        if decompressor is None:
            if '.zip' in url:
                kwargs['processor'] = pooch.processors.Unzip(extract_dir=path)
            elif '.tar' in url:
                kwargs['processor'] = pooch.processors.Untar(extract_dir=path)
            elif ('.gz' in url) or ('.bz2' in url) or ('.xz' in url):
                kwargs['processors'] = pooch.processors.Decompress(extract_dir=path)
    
    props = dict(fname=fname, path=path)
    props.update(kwargs)
    
    # here we do the actual downloading
    flist = pooch.retrieve(url, None, **props)

    change_file_permissions(flist, premission)
    
    # return the string if it's the only item in the list
    if isinstance(flist, list):
        if len(flist) == 1:
            return flist[0]
    else:
        return flist


def get_flist_from_url(
    url,
    username=None,
    password=None,
    use_cache=False,
    cache_path="./_urls.cache",
    **kwargs,
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
    from pathlib import Path as posixpath
    from urllib.parse import urlparse
    import fsspec

    if "*" not in url:
        return [url]

    if use_cache:
        cache_path = posixpath(cache_path)
        if cache_path.is_file():
            with open(cache_path, "r") as file:
                flist = file.read().split("\n")
            return sorted(flist)

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
