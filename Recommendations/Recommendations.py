import os
from tqdm import tqdm

from pyarrow.parquet import ParquetFile
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

import pandas as pd

class Recommendations:
    def __init__(self, Parquet_Data_dir):
        self._Parquet_Data_dir = Parquet_Data_dir
        self._model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=1)
        self._model_Trained = False
        self._train_matrix = None
        self._MapData = {}

    def MapData(self, Track_Data):
        # if self._MapData.keys() == 
        def Mapping(SeriesData, Message):
            counter = 0
            data_to_int, int_to_data = {}, {}
            unique_data = SeriesData.unique()
            with tqdm(total=len(unique_data), desc=Message) as pbar:
                for data in unique_data:
                    if data not in data_to_int:
                        data_to_int[data] = counter
                        int_to_data[counter] = data
                        counter += 1
                    pbar.update(1)
            return data_to_int, int_to_data

        user_to_int, int_to_user = Mapping(Track_Data["user_id"], "Mapping Users")
        song_to_int, int_to_song = Mapping(Track_Data["song_id"], "Mapping Songs")
        self._MapData["user_to_int"] = user_to_int
        self._MapData["int_to_user"] = int_to_user
        self._MapData["song_to_int"] = song_to_int
        self._MapData["int_to_song"] = int_to_song

    def precision_at_k(self, test_data : pd.DataFrame, user_item_matrix, k=10):
        hits = 0
        total = 0

        for user in test_data["user_id"].unique():
            test_tracks = set(test_data[test_data["user_id"] == user]["song_id"])
            print(len(self._model.recommend(user, user_item_matrix[user], N=k)))

            recommended_tracks = [
                track for track, _ in self._model.recommend(user, user_item_matrix[user], N=k)
            ]

            hits += len(set(recommended_tracks) & test_tracks)
            total += len(test_tracks)

        return hits / total * 100

    def fit(self):
        trainTriplits_pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "train_triplets.parquet"))
        Track_Data : pd.DataFrame = trainTriplits_pf.read().to_pandas()
        if self._MapData == {}:
            self.MapData(Track_Data)
        Track_Data["user_id"] = Track_Data["user_id"].map(self._MapData["user_to_int"])
        Track_Data["song_id"] = Track_Data["song_id"].map(self._MapData["song_to_int"])
        train_data, test_data = train_test_split(Track_Data, test_size=0.2, random_state=42)
        Track_Data = None

        train_matrix = sp.coo_matrix((train_data["play_count"], (train_data["user_id"], train_data["song_id"]))).tocsr()
        print("Fitting Model")
        self._model.fit(train_matrix, show_progress=True)
        precision = self.precision_at_k(test_data, train_matrix)
        print(f"Precision at k: {precision:.2f}/100")

        return train_matrix

    def get_similar_tracks(self, user_id, num_of_tracks=10):
        if self._train_matrix is None:
            self._train_matrix = self.fit()
        user_items = self._train_matrix[self._MapData["user_to_int"][user_id]]

        scores = self._model.recommend(user_id, user_items, N=num_of_tracks)

        recommendations = pd.DataFrame(scores, columns=["song_id", "likelihood"])
        recommendations["song_id"] = recommendations["song_id"].map(self._MapData["int_to_song"])
        return recommendations

    def get_common_listened_tracks(self):
        pass
