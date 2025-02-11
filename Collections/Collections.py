import os, difflib
from pyarrow.parquet import ParquetFile
import pandas as pd
from tqdm import tqdm

from nltk.corpus import wordnet
from nltk import download as nltk_download
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker


def get_word_themes(word):
    stemmer = PorterStemmer()
    word = stemmer.stem(word)
    spell = SpellChecker()
    word = spell.correction(word)
    synsets = wordnet.synsets(word)

    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemma_names():
            synonyms.add(lemma)

    return list(synonyms)

class Collections:
    def __init__(self, Parquet_Data_dir, debug=False):
        self._debug = debug
        self._Parquet_Data_dir = Parquet_Data_dir
        nltk_download('wordnet')

    def Get_tracks_data(self, tracks_ID_df):
        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "Merged_tracks_data.parquet"))
        Merge_DataFrame = pf.read().to_pandas()
        columns_to_return = ["artist_name", "title", "play_count"]
        DataFrame = pd.DataFrame(columns=columns_to_return)
        filtered = Merge_DataFrame[Merge_DataFrame["track_id"].isin(tracks_ID_df["track_id"])]
        if not filtered.empty:
            DataFrame = pd.concat([DataFrame, filtered[columns_to_return]], axis=0)
        if len(DataFrame) != len(tracks_ID_df):
            raise ValueError("Something went wrong")

        DataFrame.sort_values(by="play_count", ascending=False, inplace=True)
        DataFrame.index.name = "index_number"
        return DataFrame

    def Get_mxm_data(self) -> pd.DataFrame:
        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "mxm_dataset_train.parquet"))
        return pf.read().to_pandas()

    def Baseline(self, theme, num_of_tracks = 100):
        theme_keywords = get_word_themes(theme)
        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "mxm_dataset_train.parquet"))

        columns_in_theme = []
        with tqdm(total=int(len(pf.schema.names)), desc=f"Finding columns for {theme} theme") as pbar:
            for col in pf.schema.names:
                matches = difflib.get_close_matches(col.lower(), theme_keywords, n=1, cutoff=0.8)
                if matches:
                    columns_in_theme.append(col)
                pbar.update(1)

        num_row_groups = pf.metadata.num_row_groups
        Scors_DataFrame = pd.DataFrame(columns=["track_id", "score"])
        with tqdm(total=int(num_row_groups), desc=f"Getting {theme} theme scores") as pbar:
            for rg in range(num_row_groups):
                tmp_df = pd.DataFrame(columns=["track_id", "score"])
                mxm_df :pd.DataFrame = pf.read_row_group(rg).to_pandas()
                tmp_df["track_id"] = mxm_df["track_id"]
                tmp_df["score"] = mxm_df[columns_in_theme].sum(axis=1)
                tmp_df.drop(tmp_df[tmp_df["score"] == 0].index, inplace=True)

                Scors_DataFrame = pd.concat([Scors_DataFrame, tmp_df], axis=0)
                pbar.update(1)

            Scors_DataFrame.sort_values(by="score", ascending=False, inplace=True)
            Scors_DataFrame.reset_index(drop=True, inplace=True)
        return self.Get_tracks_data(Scors_DataFrame.head(num_of_tracks))

    def Word2Vec(self):
        pass

    def Classification(self):
        pass
