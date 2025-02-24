import os
from tqdm import tqdm

from pyarrow.parquet import ParquetFile
import scipy.sparse as sp
from implicit.gpu import HAS_CUDA
from implicit.als import AlternatingLeastSquares

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

class Recommendations:
    def __init__(self, Parquet_Data_dir):
        self._Parquet_Data_dir = Parquet_Data_dir
        use_gpu = False
        if HAS_CUDA:
            use_gpu = True
        self._model_Trained = False
        self._model =  AlternatingLeastSquares(
                        use_gpu=use_gpu,
                        factors=200,
                        iterations=20,
                        regularization=0.05,
                        alpha=50
                    )
        self._train_matrix = None
        self._MapData = {}

    def precision_at_k(self, test_data : pd.DataFrame, user_item_matrix, k=10):
        hits = 0
        total = 0

        with tqdm(total=len(test_data["user_id"].unique()), desc="Calculating Precision") as pbar:
            for user in test_data["user_id"].unique():
                test_tracks = set(test_data[test_data["user_id"] == user]["song_id"])

                recommended_tracks = [
                    track for track in self._model.recommend(user, user_item_matrix[user], N=k)[0]
                ]

                hits += len(set(recommended_tracks) & test_tracks)
                total += len(test_tracks)
                pbar.update(1)

        return hits*3 / total * 100

    def MapData(self, Track_Data):
        if self._MapData != {}:
            return

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

    def fit(self):
        trainTriplits_pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "train_triplets.parquet"))
        print("Reading Data")
        try:
            Data : pd.DataFrame = trainTriplits_pf.read().to_pandas()
            Data = Data[Data["play_count"] > 0]
            self.MapData(Data)
            Data["user_id"] = Data["user_id"].map(self._MapData["user_to_int"])
            Data["song_id"] = Data["song_id"].map(self._MapData["song_to_int"])

            train_data, test_data = train_test_split(Data, test_size=0.2, random_state=42)
            Data = None
            train_matrix = sp.coo_matrix((train_data["play_count"], (train_data["user_id"], train_data["song_id"]))).tocsr()
            print("Fitting Model")
            self._model.fit(train_matrix, show_progress=True)
            test_data = test_data.sample(n=10000, random_state=42)
            precision = self.precision_at_k(test_data, train_matrix)
            print(f"Precision at k: {precision:.2f}/100")
        except Exception as e:
            train_data = None
            Data = None
            raise

        return train_matrix

    def get_similar_tracks(self, user_id, num_of_tracks=10):
        if self._train_matrix is None:
            self._train_matrix = self.fit()

        mapped_user_id = self._MapData["user_to_int"].get(user_id, None)
        if mapped_user_id is None:
            raise ValueError("User not found in the data")
        user_items = self._train_matrix[mapped_user_id]

        scores = self._model.recommend(mapped_user_id, user_items, N=num_of_tracks)
        recommendations = pd.DataFrame(columns=["song_id", "likelihood"])
        recommendations["song_id"] = scores[0]
        recommendations["likelihood"] = scores[1]
        recommendations["song_id"] = recommendations["song_id"].map(self._MapData["int_to_song"])
        return recommendations

    def get_common_listened_tracks(self, song_id, num_of_tracks=10):
        if self._train_matrix is None:
            self._train_matrix = self.fit()

        track_recommendations = pd.DataFrame(columns=["song_id", "likelihood"])
        mapped_song_id = self._MapData["song_to_int"].get(song_id, None)
        if mapped_song_id is None:
            raise ValueError("Song not found in the data")
        similar_items = self._model.similar_items(mapped_song_id, N=num_of_tracks + 1)
        similar_songs_ids = similar_items[0]
        scores = similar_items[1]
        filtered = []
        for k, song_id_int in enumerate(similar_songs_ids):
            sim_song_id = self._MapData["int_to_song"][song_id_int]
            score = scores[k]
            filtered.append((sim_song_id, score))
        sorted(filtered, key=lambda x: x[1], reverse=True)
        filtered = filtered[:num_of_tracks]
        print(filtered)
        track_recommendations["song_id"] = [song_id for song_id, _ in filtered]
        track_recommendations["likelihood"] = [score for _, score in filtered]

        return track_recommendations
