import pandas as pd

class Classifier:
    def __init__(self, Parquet_Data_dir, debug=False):
        self._debug = debug
        self._Parquet_Data_dir = Parquet_Data_dir
        self._num_of_tracks_for_each_label = 500
        self._Labeled_tracks = None

    def Label_Data(self, labels, Word2Vec : callable):
        if self._Labeled_tracks is not None:
            return
        columns = ["track_id", "label"]
        labeled_df = pd.DataFrame(columns=columns)
        for label in labels:
            data = Word2Vec(label, num_of_tracks=self._num_of_tracks_for_each_label, track_id=True)
            data["label"] = label
            labeled_df = pd.concat([labeled_df, data[columns]], axis=0)

        self._Labeled_tracks = labeled_df
