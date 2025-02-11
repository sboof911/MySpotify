from pyarrow import string, int64
from .utils import Convert_file_To_Parquet, prepare_mxm_dataset_train, os

def Convert_msd_to_parquet(Data_folder, output_dir, fileNames, buffer_size=10000):
    file_path = ""
    for fileName in fileNames:
        if "msd" in fileName:
            file_path = os.path.join(Data_folder, fileName)
    if file_path == "":
        raise ValueError("No MSD file found")

    kwargs = {
        "file_path": file_path,
        "output_dir": output_dir,
        "numOfLines": 280831,
        "columns": [("track_id", string()),
                    ("majority_genre", string()),
                    ("minority_genre", string())],
        "sep": "\t",
        "buffer_size":buffer_size
    }
    Convert_file_To_Parquet(**kwargs)

def Convert_Triplets_to_parquet(Data_folder, output_dir, fileNames, buffer_size=10000):
    file_path = ""
    for fileName in fileNames:
        if "triplets" in fileName:
            file_path = os.path.join(Data_folder, fileName)
    if file_path == "":
        raise ValueError("No TRIPLETS file found")

    kwargs = {
        "file_path": file_path,
        "output_dir": output_dir,
        "numOfLines": 48373586,
        "columns": [
            ("user_id", string()),
            ("song_id", string()),
            ("play_count", int64())
        ],
        "sep": "\t",
        "buffer_size":buffer_size
    }
    Convert_file_To_Parquet(**kwargs)

def Convert_unique_tracks_to_parquet(Data_folder, output_dir, fileNames, buffer_size=10000):
    file_path = ""
    for fileName in fileNames:
        if "unique" in fileName:
            file_path = os.path.join(Data_folder, fileName)
    if file_path == "":
        raise ValueError("No UNIQUE_TRACKS file found")

    kwargs = {
        "file_path": file_path,
        "output_dir": output_dir,
        "numOfLines": 1000000,
        "columns": [
            ("track_id", string()),
            ("song_id", string()),
            ("artist_name", string()),
            ("title", string())
        ],
        "sep": "<SEP>",
        "buffer_size":buffer_size
    }
    Convert_file_To_Parquet(**kwargs)

def Convert_mxm_to_parquet(Data_folder, output_dir, fileNames, buffer_size=10000):
    file_path = ""
    for fileName in fileNames:
        if "mxm" in fileName:
            file_path = os.path.join(Data_folder, fileName)
    if file_path == "":
        raise ValueError("No MXM file found")

    kwargs = {
        "file_path": file_path,
        "output_dir": output_dir,
        "numOfLines": 210519,
        "buffer_size":buffer_size
    }
    prepare_mxm_dataset_train(**kwargs)
