import os, difflib
from pyarrow.parquet import ParquetFile
import pyarrow.compute as pc
import pandas as pd

def Get_top_Tracks_SongIds(Data_folder, num_of_tracks):
    Merged_Song = ParquetFile(os.path.join(Data_folder, "Merged_tracks_data.parquet"))
    table = Merged_Song.iter_batches(batch_size=num_of_tracks)

    DataFrame = next(table).to_pandas()
    DataFrame.index.name = "index_number"
    columns = ["artist_name", "title", "play_count"]
    return DataFrame[columns]

def Get_Top_Tracks(Data_folder, num_of_tracks):
    Top_song_ids = Get_top_Tracks_SongIds(Data_folder, num_of_tracks)

    return Top_song_ids

def detect_genre(Data_folder, input_genre):
    pf = ParquetFile(os.path.join(Data_folder, "Merged_tracks_data.parquet"))
    df : pd.DataFrame = pf.read().to_pandas()
    allowed_genres = df["majority_genre"].drop_duplicates().tolist()
    allowed_genres_lower_map = {g.lower(): g for g in allowed_genres if g}
    allowed_genres_lower = list(allowed_genres_lower_map.keys())

    matches = difflib.get_close_matches(input_genre.lower(), allowed_genres_lower, n=1, cutoff=0.8)
    if matches:
        return allowed_genres_lower_map[matches[0]]
    else:
        return ""

def Get_TopTracks_ByGenre(Data_folder, num_of_tracks, genre, drop_genre=True):
    def drop_majority_genre(table):
        if drop_genre:
            return table.drop(columns="majority_genre")
        return table
    Merged_tracks = ParquetFile(os.path.join(Data_folder, "Merged_tracks_data.parquet"))
    genre = detect_genre(Data_folder, genre)
    if genre == "":
        raise ValueError(f"Genre {genre} not found")
    num_row_groups = Merged_tracks.metadata.num_row_groups
    tracks_df = pd.DataFrame(columns=["track_id", "title", "play_count"])
    for rg in range(num_row_groups):
        columns = tracks_df.columns.tolist()
        columns.append("majority_genre")
        table = Merged_tracks.read_row_group(rg, columns=columns)
        mask = pc.equal(table["majority_genre"], genre)
        filtered_table = table.filter(mask).to_pandas()
        if len(filtered_table) < num_of_tracks-len(tracks_df):
            tracks_df = pd.concat([tracks_df, drop_majority_genre(filtered_table)], axis=0)
        else:
            tracks_df = pd.concat([tracks_df, drop_majority_genre(filtered_table.head(num_of_tracks-len(tracks_df)))], axis=0)
            break
    tracks_df.index.name = "index_number"
    return tracks_df