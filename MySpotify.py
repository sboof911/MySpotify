import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
from tqdm import tqdm
import os
from utils.readfiles import Convert_To_Parquet, DATA_FOLDER
from utils.Top_Tracks import get_best_tracks_df
from utils.Top_Tracks_Genre import get_best_tracks_by_genre

class MySpotify:
    fileNames = ["p02_msd_tagtraum_cd2", "mxm_dataset_train", "p02_unique_tracks", "train_triplets"]
    def __init__(self, path : str):
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not os.path.exists(path):
            raise ValueError("path does not exist")
        if not os.path.isdir(path):
            raise ValueError("path must be a directory")
        self._path = path
        self._chunk_size = 10000

    def PrepareData(self):
        Convert_To_Parquet(self._path, self._chunk_size)
        self.getData()

    def getData(self):
        def get_file_path(file_name):
            file_path = os.path.join(self._path, f"{DATA_FOLDER}/{file_name}.parquet")
            if not os.path.exists(file_path):
                raise ValueError(f"{file_name} not found!!")
            return file_path

        self._msdData = pq.ParquetFile(get_file_path("p02_msd_tagtraum_cd2"))
        self._mxmData = pq.ParquetFile(get_file_path("mxm_dataset_train"))
        self._uniqueTracks = pq.ParquetFile(get_file_path("p02_unique_tracks"))
        self._trainTriplets = pq.ParquetFile(get_file_path("train_triplets"))

    # Get the Top Tracks Dataframe:

    def get_Top_Tracks(self, numoftracks=250):
        return get_best_tracks_df(self._uniqueTracks, self._trainTriplets, numoftracks)

    ################################################################################################
    # Top tracks by genre

    def get_Top_Tracks_By_Genre(self, genre : str="Rock", numoftracks=100):
        return get_best_tracks_by_genre(self._uniqueTracks, self._trainTriplets, self._msdData, numoftracks, genre)

    ################################################################################################
    # Cleanup
    def cleanup(self):
        self._msdData = None
        self._mxmData = None
        self._uniqueTracks = None
        self._trainTriplets = None

    ################################################################################################
    @property
    def path(self):
        return self._path
