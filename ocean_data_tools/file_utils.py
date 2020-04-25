def unzip(zip_path, dest_dir=None, verbose=1):
    """returns a list of unzipped file names"""
    import os
    from zipfile import ZipFile

    def get_destination_directory(zipped):
        file_name = zipped.filename
        file_list = zipped.namelist()
        if len(file_list) == 1:
            destdir = os.path.split(file_name)[0]
        else:
            destdir = os.path.splitext(file_name)[0]

        return destdir

    def get_list_of_zipped_files(zipped, dest_dir):
        flist_zip = set(zipped.namelist())
        flist_dir = set(os.listdir(dest_dir))

        for file in flist_dir:
            if not is_local_file_valid(file):
                flist_dir -= set(file)

        files_to_extract = list(flist_zip - flist_dir)

        if not files_to_extract:
            if verbose:
                print(f"All files extracted: {zipped.filename}")
        return files_to_extract

    if not os.path.isfile(zip_path):
        raise OSError(f"The zip file does not exist: {zip_path}")

    zipped = ZipFile(zip_path, "r")
    if dest_dir is None:
        dest_dir = get_destination_directory(zipped)
        os.makedirs(dest_dir, exist_ok=True)

    files_to_extract = get_list_of_zipped_files(zipped, dest_dir)
    for file in files_to_extract:
        if verbose:
            print(f" Extracting: {file}")
        zipped.extractall(path=dest_dir, members=[file])

    return [os.path.join(dest_dir, f) for f in zipped.namelist()]


def gunzip(zip_path, dest_path=None):
    import shutil
    import gzip

    if dest_path is None:
        dest_path = zip_path.replace(".gz", "")

    with gzip.open(zip_path, "rb") as f_in:
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
            return f_out


def untar(tar_path, dest_dir=None, verbose=1):
    """returns a list of untarred file names"""
    import os
    import pathlib
    import tarfile

    if not os.path.isfile(tar_path):
        raise OSError(f"The tar file does not exist: {tar_path}")

    if tar_path.endswith("gz"):
        mode = "r:gz"
    else:
        mode = "r:"
    tarred = tarfile.open(tar_path, mode)

    if dest_dir is None:
        dest_dir = pathlib.Path(tar_path).parent
    else:
        os.makedirs(dest_dir, exist_ok=True)

    tarred.extractall(path=dest_dir)

    return [os.path.join(dest_dir, f) for f in tarred.getnames()]


def is_local_file_valid(local_path):
    from os.path import isfile

    if not isfile(local_path):
        return False

    # has an opener been passed, if not assumes file is valid
    if local_path.endswith(".nc"):
        from netCDF4 import Dataset as opener

        error = OSError
    elif local_path.endswith(".zip"):
        from zipfile import ZipFile as opener, BadZipFile as error
    else:
        error = BaseException

        def opener(p):
            return None  # dummy opener

    # tries to open the path, if it fails, not valid, if it passes, valid
    try:
        with opener(local_path):
            return True
    except error:
        return False
