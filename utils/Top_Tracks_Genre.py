from tqdm import tqdm
import pyarrow.compute as pc
from pyarrow import array
import pandas as pd
from pyarrow.parquet import ParquetFile

def HandleErrors(numoftracks, genre, trainTriplets, uniqueTracks, msdData):
    if not isinstance(numoftracks, int):
        raise TypeError("numoftracks must be an integer")
    if numoftracks < 1:
        raise ValueError("numoftracks must be greater than 0")
    if not isinstance(genre, str):
        raise TypeError("genre must be a string")
    if not genre:
        raise ValueError("genre must not be empty")
    if not isinstance(trainTriplets, ParquetFile):
        raise TypeError("trainTriplets must be a ParquetFile")
    if not isinstance(uniqueTracks, ParquetFile):
        raise TypeError("uniqueTracks must be a ParquetFile")
    if not isinstance(msdData, ParquetFile):
        raise TypeError("msdData must be a ParquetFile")


def get_all_tracks_Genre(msdData : ParquetFile, genre : str):
    num_row_groups = msdData.metadata.num_row_groups
    tracks = []
    with tqdm(total=int(num_row_groups), desc="Finding tracks by genre") as pbar:
        for rg in range(num_row_groups):
            table = msdData.read_row_group(rg, columns=["track_id", "majority_genre", "minority_genre"])
            mask_major = pc.equal(pc.ascii_lower(table["majority_genre"]), genre.lower())
            mask_minor = pc.equal(pc.ascii_lower(table["minority_genre"]), genre.lower())
            mask = pc.or_(mask_major, mask_minor)
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                filtered_dict = filtered.to_pydict()
                tracks.extend(filtered_dict["track_id"])
            pbar.update(1)
    
    print(f"Found {len(tracks)} tracks with genre {genre}")

    return set(tracks)

def get_tracks_genre_data_by_id(uniqueTracks : ParquetFile, Track_ids : set):
    num_row_groups = uniqueTracks.metadata.num_row_groups
    Tracks_data = []

    with tqdm(total=int(num_row_groups), desc=f"Finding tracks song_id by track_id") as pbar:
        for rg in range(num_row_groups):
            table = uniqueTracks.read_row_group(rg, columns=["track_id", "song_id", "artist_name", "title"])
            mask = pc.is_in(table["track_id"], value_set=array(list(Track_ids)))
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                filtered_dict : dict = filtered.to_pydict()
                for i in range(filtered.num_rows):
                    row = {key: filtered_dict[key][i] for key in filtered_dict.keys()}
                    Tracks_data.append(row)
                    Track_ids.remove(filtered_dict["track_id"][i])

            pbar.update(1)

        for missing in Track_ids:
            print(f"Track ID | {missing} | not found in unique tracks")

    return Tracks_data

def get_top_tracks_by_genre(trainTriplets: ParquetFile, Tracks_data : list, numoftracks: int):
    num_row_groups = trainTriplets.metadata.num_row_groups
    best_tracks = []
    tracks_map = {track["song_id"]: track for track in Tracks_data}

    with tqdm(total=int(num_row_groups), desc="Finding best tracks by genre") as pbar:
        for rg in range(num_row_groups):
            table = trainTriplets.read_row_group(rg, columns=["song_id", "play_count"])
            mask = pc.is_in(table["song_id"], value_set=array([track["song_id"] for track in Tracks_data]))
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                filtered_dict = filtered.to_pydict()
                for i in range(filtered.num_rows):
                    if len(best_tracks) <= numoftracks \
                    or filtered_dict["play_count"][i] > best_tracks[numoftracks-1]["play_count"]:

                        song_id = filtered_dict["song_id"][i]
                        if song_id in tracks_map:
                            track_data = tracks_map.pop(song_id)
                        else:
                            pass
                        track_data["play_count"] = filtered_dict["play_count"][i]
                        if len(best_tracks) < numoftracks:
                            best_tracks.append(track_data)
                        else:
                            best_tracks[numoftracks-1] = track_data
                        best_tracks.sort(key=lambda x: x["play_count"], reverse=True)

            pbar.update(1)
        
    for missing in tracks_map.keys():
        print(f"Track ID | {missing} | play_count not found in Triplets")

    return pd.DataFrame(best_tracks)

def get_best_tracks_by_genre(uniqueTracks : ParquetFile, trainTriplets : ParquetFile, msdData : ParquetFile, numoftracks : int, genre : str):
    HandleErrors(numoftracks, genre, trainTriplets, uniqueTracks, msdData)

    Tracks = get_all_tracks_Genre(msdData, genre)
    if not Tracks:
        return pd.DataFrame(columns=["track_id", "song_id", "artist_name", "title", "play_count"])
    Tracks_data = get_tracks_genre_data_by_id(uniqueTracks, Tracks)
    Tracks_data = get_top_tracks_by_genre(trainTriplets, Tracks_data, numoftracks)
    return Tracks_data
