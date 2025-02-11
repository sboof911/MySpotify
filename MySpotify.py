import os

from Collections.Collections import Collections

from PrepocessData.ReadZip import upzip_data
from PrepocessData.ConvertFiles import Convert_msd_to_parquet, Convert_Triplets_to_parquet, Convert_unique_tracks_to_parquet, Convert_mxm_to_parquet
from PrepocessData.MergeData import Merge_All_Data

from TopTracks.Top_Listen_Tracks import Get_Top_Tracks, Get_TopTracks_ByGenre
DATA_FOLDER = "data/"
PARQUET_FOLDER = "parquet/"

class MySpotify(Collections):
    fileNames = ["msd_tagtraum_cd2.cls", "mxm_dataset_train.txt", "unique_tracks.txt", "train_triplets.txt"]

    def __init__(self, zipFile_abspath : str = "", debug=False):
        self._Data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_FOLDER)
        if not os.path.exists(self._Data_folder):
            if not os.path.exists(zipFile_abspath):
                raise ValueError("No zip file provided and not data folder found")
            else:
                upzip_data(zipFile_abspath, self._Data_folder, self.fileNames)

        if any([not os.path.exists(os.path.join(self._Data_folder, file)) for file in self.fileNames]):
            upzip_data(zipFile_abspath, self._Data_folder, self.fileNames)

        super().__init__(os.path.join(self._Data_folder, PARQUET_FOLDER), debug)
        print("All files are present!")

    def convert_files(self):
        if not os.path.exists(self._Parquet_Data_dir):
            os.makedirs(self._Parquet_Data_dir, exist_ok=True)
        kwargs = {
            "Data_folder": self._Data_folder,
            "output_dir": self._Parquet_Data_dir,
            "fileNames": self.fileNames,
            "buffer_size": 10000
        }
        Convert_msd_to_parquet(**kwargs)
        Convert_Triplets_to_parquet(**kwargs)
        Convert_unique_tracks_to_parquet(**kwargs)
        ######
        Convert_mxm_to_parquet(**kwargs)
        print("All files converted to parquet!!")

    def PreProcess_Data(self):
        Merge_All_Data(self._Parquet_Data_dir)

    def get_Top_Tracks(self, num_of_tracks):
        if not os.path.exists(self._Parquet_Data_dir):
            raise ValueError("No parquet data found")

        return Get_Top_Tracks(self._Parquet_Data_dir, num_of_tracks)

    def Get_TopTracks_By_Genre(self, num_of_tracks, genre):
        if not os.path.exists(self._Parquet_Data_dir):
            raise ValueError("No parquet data found")

        return Get_TopTracks_ByGenre(self._Parquet_Data_dir, num_of_tracks,
                                     genre, not self._debug)
        