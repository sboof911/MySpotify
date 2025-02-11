import zipfile, os

def unzip_file(zip_file, output_dir):
    print(f"Unzipping {zip_file} to {output_dir}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def movefiles(output_dir):
    zip_output_dir = os.path.join(output_dir, "P02. MySpotify")
    for file in os.listdir(zip_output_dir):
        src_path = os.path.join(zip_output_dir, file)
        file = file.replace("p02_", "")
        dst_path = os.path.join(output_dir, file)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.rename(src_path, dst_path)
    os.removedirs(zip_output_dir)

def upzip_data(zip_file, output_dir, filenames):
    unzip_file(zip_file, output_dir)
    movefiles(output_dir)
    for file in os.listdir(output_dir):
        if file.endswith(".zip"):
            zip_file_path = os.path.join(output_dir, file)
            unzip_file(zip_file_path, output_dir)
            if os.path.exists(zip_file_path[:-4]):
                os.remove(zip_file_path)

    for file in filenames:
        if not os.path.exists(os.path.join(output_dir, file)):
            raise ValueError(f"File {file} not found in {output_dir}")
