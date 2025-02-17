import os, difflib
from pyarrow.parquet import ParquetFile, ParquetWriter
from pyarrow import Table
import pandas as pd
import numpy as np
from tqdm import tqdm

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import download as nltk_download
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from.Classifier import Classifier

def filter_words(words):
    stop_words = set(stopwords.words("english"))
    filtered = []

    for word in words:
        clean_word = word.lower().strip()

        if clean_word.isalpha() and clean_word not in stop_words:
            filtered.append(word)
    return filtered

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

class Collections(Classifier):
    def __init__(self, Parquet_Data_dir, debug=False):
        self._debug = debug
        self._Parquet_Data_dir = Parquet_Data_dir
        nltk_download('wordnet')
        nltk_download("stopwords")
        super().__init__(Parquet_Data_dir, debug)

    def Get_tracks_data(self, tracks_ID_df, track_id):
        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "Merged_tracks_data.parquet"))
        Merge_DataFrame = pf.read().to_pandas()
        columns_to_return = ["artist_name", "title", "play_count"]
        if track_id:
            columns_to_return.append("track_id")
        DataFrame = pd.DataFrame(columns=columns_to_return)
        filtered = Merge_DataFrame[Merge_DataFrame["track_id"].isin(tracks_ID_df["track_id"])]
        if not filtered.empty:
            DataFrame = pd.concat([DataFrame, filtered[columns_to_return]], axis=0)
        if len(DataFrame) != len(tracks_ID_df):
            raise ValueError("Something went wrong")

        DataFrame.sort_values(by="play_count", ascending=False, inplace=True)
        DataFrame["index_number"] = range(0, len(DataFrame))
        return DataFrame

    def Get_mxm_data(self) -> pd.DataFrame:
        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "mxm_dataset_train.parquet"))
        return pf.read().to_pandas()

    def get_scores(self, pf, theme, words_in_theme, num_of_tracks, track_id=False):
        num_row_groups = pf.metadata.num_row_groups
        Scors_DataFrame = pd.DataFrame(columns=["track_id", "score"])
        with tqdm(total=int(num_row_groups), desc=f"Getting {theme} theme scores") as pbar:
            for rg in range(num_row_groups):
                tmp_df = pd.DataFrame(columns=["track_id", "score"])
                mxm_df :pd.DataFrame = pf.read_row_group(rg).to_pandas()
                tmp_df["track_id"] = mxm_df["track_id"]
                tmp_df["score"] = mxm_df[words_in_theme].sum(axis=1)
                tmp_df.drop(tmp_df[tmp_df["score"] == 0].index, inplace=True)

                Scors_DataFrame = pd.concat([Scors_DataFrame, tmp_df], axis=0)
                pbar.update(1)

            Scors_DataFrame.sort_values(by="score", ascending=False, inplace=True)
            Scors_DataFrame.reset_index(drop=True, inplace=True)
        return self.Get_tracks_data(Scors_DataFrame.head(num_of_tracks), track_id)

    def Baseline(self, theme, num_of_tracks = 100):
        theme_keywords = get_word_themes(theme)
        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "mxm_dataset_train.parquet"))

        words_in_theme = []
        with tqdm(total=int(len(pf.schema.names[2:])), desc=f"Finding words for {theme} theme") as pbar:
            for word in pf.schema.names[2:]:
                matches = difflib.get_close_matches(word.lower(), theme_keywords, n=1, cutoff=0.8)
                if matches:
                    words_in_theme.append(word)
                pbar.update(1)

        return self.get_scores(pf, theme, words_in_theme, num_of_tracks)

    def Word2Vec(self, theme, num_of_tracks = 100, track_id=False):
        theme_keywords = get_word_themes(theme)
        pf = ParquetFile(os.path.join(self._Parquet_Data_dir, "mxm_dataset_train.parquet"))

        words = filter_words(pf.schema.names[2:])
        word_sentences = [words]
        w2v_model = Word2Vec(word_sentences, vector_size=100, window=5, min_count=1, workers=4)

        expanded_keywords = set(theme_keywords)
        for keyword in theme_keywords:
            if keyword in w2v_model.wv:
                similar_words = [word for word, _ in w2v_model.wv.most_similar(keyword, topn=5)]
                expanded_keywords.update(similar_words)

        if len(expanded_keywords) == 0:
            raise ValueError("No keywords found")

        columns_in_theme = [word for word in expanded_keywords if word in words]

        return self.get_scores(pf, theme, columns_in_theme, num_of_tracks, track_id)

    def Classification(self, labels, theme, num_of_tracks=100):
        self.Label_Data(labels, self.Word2Vec)

        return None