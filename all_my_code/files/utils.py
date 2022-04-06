        
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


def give_group_permission(path, permission=774):
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
            