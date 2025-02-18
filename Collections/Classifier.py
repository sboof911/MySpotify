import os
import pandas as pd
from tqdm import tqdm
from pyarrow.parquet import ParquetFile

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class Classifier:
    def __init__(self, Parquet_Data_dir, debug=False):
        self._debug = debug
        self._Parquet_Data_dir = Parquet_Data_dir
        self._num_of_tracks_for_each_label = 500
        self._Labeled_tracks = None
        self._tfidf = TfidfTransformer()
        self._label_encoder = LabelEncoder()
        self._model = None
        self._call_filter_words : callable = None
        self._filtred_columns = None

    def Label_Data(self, labels, Word2Vec : callable):
        if self._Labeled_tracks is not None:
            print("Data is already labeled")
            return
        columns = ["track_id", "theme_label"]
        labeled_df = pd.DataFrame(columns=columns)
        for label in labels:
            data = Word2Vec(label, num_of_tracks=self._num_of_tracks_for_each_label, track_id_verbose=True)
            data["theme_label"] = label
            labeled_df = pd.concat([labeled_df, data[columns]], axis=0)

        self._Labeled_tracks = labeled_df

    def Merge_Lyrics_with_Labels(self):
        if self._Labeled_tracks is None:
            raise Exception("You need to label the data first")

        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "mxm_dataset_train.parquet"))
        num_row_groups = pf.metadata.num_row_groups
        columns = pf.metadata.schema.names
        columns.append("theme_label")
        Data = pd.DataFrame(columns=columns)
        with tqdm(total=int(num_row_groups), desc=f"Merging lyrics with them theme label") as pbar:
            for rg in range(num_row_groups):
                mxm_df : pd.DataFrame = pf.read_row_group(rg).to_pandas()
                merged = pd.merge(mxm_df, self._Labeled_tracks, on="track_id", how="inner")
                Data = pd.concat([Data, merged], axis=0)
                pbar.update(1)
        return Data.drop(columns=["mxm_track_id"])

    def Filter_Lyrics(self, Data : pd.DataFrame, filter_words : callable):
        if self._call_filter_words is None:
            self._call_filter_words = filter_words

        current_columns = Data.columns.drop("track_id")
        filtered_columns = set(self._call_filter_words(current_columns))
        filtered_columns.add("theme_label")
        filtered_columns = list(filtered_columns)
        self._filtred_columns = filtered_columns
        Data = Data.loc[:, filtered_columns]
        Data.fillna(0, inplace=True)

        labeled_Data = Data["theme_label"]
        word_columns = Data.columns.drop("theme_label")
        Data = pd.DataFrame(self._tfidf.fit_transform(Data[word_columns]).toarray())
        Data["theme_label"] = self._label_encoder.fit_transform(labeled_Data)
        return Data

    def fit(self, Data: pd.DataFrame):
        print("Fitting the model")
        X = Data.loc[:, Data.columns.drop("theme_label")]
        y = Data["theme_label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        self._model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=1000, random_state=42)
        self._model.fit(X_train, y_train)

        y_pred_mlp = self._model.predict(X_test)

        print(f"MLP Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
        print(classification_report(y_test, y_pred_mlp, target_names=self._label_encoder.classes_))

    def predict_theme(self, song_features : pd.DataFrame):
        Prediction_data = pd.DataFrame(song_features["track_id"], columns=["track_id"])
        columns = self._filtred_columns.copy()
        columns.remove("theme_label")
        song_features = song_features.loc[:, columns]
        song_features = self._tfidf.transform(song_features)
        prediction = self._model.predict(song_features)[0]
        Prediction_data["theme_label"] = self._label_encoder.inverse_transform([prediction])[0]
        return Prediction_data
