        
def zip_folder(input_dir, output_file):
    import shutil
    shutil.make_archive(
        output_file.replace('.zip', ''),
        'zip',
        input_dir,)


def move_file_to_parent(fname, levels=1):
    """
    Move a file to the parent directory of the current directory

    Used to move files from a zipped folder to the parent directory
    """
    import os 
    from pathlib import Path as posixpath

    old_path = posixpath(fname)

    name = old_path.name
    new_path = old_path.parent.parent / name
    
    old_path.rename(new_path)
    
    return str(new_path)


def change_file_permissions(path, permission=774):
    """ 
    Give group permission to a file or directory 
    """
    from numpy import ndarray
    import os
    from pathlib import Path as posixpath
    perm = int(str(permission), base=8)

    if isinstance(path, str):
        for root, dirs, files in os.walk(path):
            [os.chmod(f, perm) for f in dirs]
            [os.chmod(f, perm) for f in files]
    elif isinstance(path, (list, tuple, ndarray)):
        for f in path:
            try:
                f = posixpath(f)
                f.chmod(perm)
                f.parent.chmod(perm)
            except:
                pass
            

def get_fnames_recursive_search(basedir, include=[], exclude=[]):
    """
    Search and match file names in a directory (recursive)
    
    Parameters
    ----------
    basedir: str (must exist as a path)
        the directory that you'd like to search through
    include: list of str
        the string patterns that must occur in the files you're looking for
    exclude: list of str
        string patterns you would like to exclude from the filenames
        
    Returns 
    -------
    A list of file names with the full path
    """
    import os
    import re

    flist = []
    for path, subdir, files in os.walk(basedir):
        for fname in files:
            if all([p in fname for p in include]):
                has_excluded = [s in fname for s in exclude]
                if not any(has_excluded):
                    flist += os.path.join(path, fname),

    flist = np.sort(flist)
    return flist