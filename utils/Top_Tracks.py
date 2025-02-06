from tqdm import tqdm
import pyarrow.compute as pc
import pandas as pd
from pyarrow.parquet import ParquetFile

def get_best_tracks_count(trainTriplets : ParquetFile, numoftracks : int):
    if not isinstance(numoftracks, int):
        raise TypeError("numoftracks must be an integer")
    if numoftracks < 1:
        raise ValueError("numoftracks must be greater than 0")

    num_row_groups = trainTriplets.metadata.num_row_groups
    best_tracks = []
    with tqdm(total=int(num_row_groups), desc="Finding best tracks id") as pbar:
        for rg in range(num_row_groups):
            table = trainTriplets.read_row_group(rg, columns=["play_count"])
            col_with_idx = list(enumerate(table["play_count"].to_pylist()))
            col_with_idx.sort(key=lambda x: x[1], reverse=True)
            for offset, val in col_with_idx:
                if len(best_tracks) < numoftracks:
                    best_tracks.append((val, rg, offset))
                    best_tracks.sort(key=lambda x: x[0], reverse=True)
                else:
                    if val > best_tracks[numoftracks-1][0]:
                        best_tracks[numoftracks-1] = (val, rg, offset)
                        best_tracks.sort(key=lambda x: x[0], reverse=True)
            pbar.update(1)

    print(f"Found top {numoftracks} rows (highest 'play_count'):")
    best_results = []

    for val, rg, offset_in_group in best_tracks:
        full_table = trainTriplets.read_row_group(rg)
        row_slice = full_table.slice(offset_in_group, 1).select(["song_id", "play_count"])
        best_results.append(row_slice.to_pydict())

    return best_results

def get_best_tracks_df(uniqueTracks : ParquetFile, trainTriplets : ParquetFile, numoftracks=250):
    best_results = get_best_tracks_count(trainTriplets, numoftracks)
    unique_best_tracks = []
    num_row_groups = uniqueTracks.metadata.num_row_groups
    for best_result in best_results:
        song_id = best_result["song_id"][0]
        play_count = best_result["play_count"][0]
        found = False
        with tqdm(total=int(num_row_groups), desc=f"Finding song ID {song_id} title and artist_name") as pbar:
            for rg in range(num_row_groups):
                table = uniqueTracks.read_row_group(rg, columns=["track_id", "song_id", "artist_name", "title"])
                mask = pc.equal(table["song_id"], song_id)
                filtered = table.filter(mask)
                if filtered.num_rows > 0:
                    filtered_dict = filtered.to_pydict()
                    filtered_dict["play_count"] = [play_count]
                    unique_best_tracks.append(filtered_dict)
                    found = True
                    pbar.update(pbar.total - pbar.n)
                    break
                pbar.update(1)
            if not found:
                print(f"Song ID | {song_id} | not found in unique tracks")

    df = pd.DataFrame(unique_best_tracks)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x[0])
    return df