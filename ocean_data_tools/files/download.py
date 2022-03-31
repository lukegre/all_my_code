def download(url, path='.', fname=None, progress=True, decompress=True, **kwargs):
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
    **kwargs: key-value
        any standard inputs of pooch 
        
    Returns
    -------
    str | list:
        if only a single entry is downloaded / decompressed, then a string will
        be returned, otherwise, a list will be returned
        
    """
    from pathlib import Path as posixpath
    import pooch
    
    if fname is None:
        fname = posixpath(url).name
        
    if progress:
        downloader = kwargs.get('downloader', None)
        if downloader is None:
            downloader = pooch.downloaders.choose_downloader(url)
        downloader.progressbar = True
        kwargs['downloader'] = downloader
    
    if decompress:
        decompressor = kwargs.get('processor', None)
        if decompressor is None:
            if '.zip' in url:
                kwargs['processor'] = pooch.processors.Unzip()
            elif '.tar' in url:
                kwargs['processor'] = pooch.processors.Untar()
            elif ('.gz' in url) or ('.bz2' in url) or ('.xz' in url):
                kwargs['processors'] = pooch.processors.Decompress()
    
    props = dict(fname=fname, path=path)
    props.update(kwargs)
    
    flist = pooch.retrieve(url, None, **props)
    
    if isinstance(flist, list):
        if len(flist) == 1:
            return flist[0]
    else:
        return flist
