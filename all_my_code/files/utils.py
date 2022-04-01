        
def zip_folder(input_dir, output_file):
    import shutil
    shutil.make_archive(
        output_file.replace('.zip', ''),
        'zip',
        input_dir,)