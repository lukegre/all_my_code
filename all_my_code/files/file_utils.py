
# FILE UTILS #
def save_nc4(ds, sname, complevel=4, dtype='float32', overwrite=True, make_parent=True):
    from pathlib import Path
    
    sname = Path(sname)

    # checking if file exists if overwrite is false
    if sname.is_file() and not overwrite:
        return
    
    # changing data type to specified type
    for key in ds.data_vars:
        ds[key] = ds[key].astype(dtype)
    
    # create compression encoding dictionary
    if complevel > 0:
        comp = dict(complevel=complevel, zlib=True)
    elif complevel == 0: 
        comp = {}
        
    # saving file with specified encoding
    ds.to_netcdf(sname, encoding={k: comp for k in ds})
        
        
def zip_folder(input_dir, output_file):
    import shutil
    shutil.make_archive(
        output_file.replace('.zip', ''),
        'zip',
        input_dir,)