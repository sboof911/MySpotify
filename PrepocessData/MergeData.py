from tqdm import tqdm
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import os
from PrepocessData.utils import get_number_of_rows

def Sum_All_Play_Counts(Data_folder, output_dir):
    trainTriplets = None
    for fileName in os.listdir(Data_folder):
        if "triplets" in fileName:
            trainTriplets = ParquetFile(os.path.join(Data_folder, fileName))
    if trainTriplets is None:
        raise ValueError("No TRIPLETS file found")

    output_path = os.path.join(output_dir, "play_count.parquet")
    if get_number_of_rows(output_path) >= 384546:
        print(f"File play_count already converted to parquet!!")
        return
    if os.path.exists(output_path):
        os.remove(output_path)

    num_row_groups = trainTriplets.metadata.num_row_groups
    data_count = {}
    with tqdm(total=int(num_row_groups), desc="Finding play count") as pbar:
        for rg in range(num_row_groups):
            table = trainTriplets.read_row_group(rg, columns=["song_id", "play_count"])
            batch_result = table.group_by("song_id").aggregate([("play_count", "sum")])
            batch_result_dict = batch_result.to_pydict()
            if len(batch_result_dict["song_id"]) != len(batch_result_dict["play_count_sum"]):
                raise ValueError("Length of song_id and play_count is not equal")
            for i in range(len(batch_result_dict["song_id"])):
                song_id = batch_result_dict["song_id"][i]
                play_count = batch_result_dict["play_count_sum"][i]
                data_count[song_id] = data_count.get(song_id, 0) + play_count
            pbar.update(1)

    data_count = [{"song_id": song_id, "play_count": play_count}
                  for song_id, play_count in data_count.items()]

    table = pa.Table.from_pylist(
            data_count,
            schema=pa.schema([("song_id", pa.string()),
                            ("play_count", pa.int64())],)
        )

    writer = None
    try:
        writer = pq.ParquetWriter(output_path, schema=table.schema)
        writer.write_table(table)
    except Exception as e:
        if writer is not None:
            writer.close()
        raise Exception(f"Error in writing data {e}")

    data_count.clear()
    print(f"Play count data prepared successfully!!")

def Merge_Song_Data(Data_folder):
    Merged_data_path = os.path.join(Data_folder, "Merged_Song_data.parquet")
    if get_number_of_rows(Merged_data_path) >= 1000000:
        print(f"File Merged_data already converted to parquet!!")
        return
    if os.path.exists(Merged_data_path):
        os.remove(Merged_data_path)

    Sum_All_Play_Counts(Data_folder, Data_folder)
    play_count_path = os.path.join(Data_folder, "play_count.parquet")
    play_count_module = ParquetFile(play_count_path)
    play_count_data = play_count_module.read()
    unique_tracks_path = os.path.join(Data_folder, "unique_tracks.parquet")
    unique_tracks = ParquetFile(unique_tracks_path)

    num_row_groups = unique_tracks.metadata.num_row_groups
    writer = None
    try:
        with tqdm(total=int(num_row_groups), desc="Merging Data") as pbar:
            for rg in range(num_row_groups):
                unique_tracks_data = unique_tracks.read_row_group(rg)
                mask = pc.is_in(play_count_data["song_id"], unique_tracks_data["song_id"])
                filtred_play_count = play_count_data.filter(mask)
                if filtred_play_count.num_rows > 0:
                    Merge_data = []
                    for row in filtred_play_count.to_pylist():
                        song_id = row["song_id"]
                        mask = pc.equal(unique_tracks_data["song_id"], song_id)
                        track_data = unique_tracks_data.filter(mask).to_pylist()
                        if len(track_data) == 0:
                            raise ValueError(f"Track data not found data found {track_data}")
                        for track in track_data:
                            Merge_data.append({"song_id": song_id, "track_id": track["track_id"],
                                            "artist_name": track["artist_name"], "title": track["title"],
                                            "play_count": row["play_count"]})

                mask_inverted = pc.is_in(unique_tracks_data["song_id"], play_count_data["song_id"])
                mask_inverted = pc.invert(mask_inverted)
                non_played_tracks = unique_tracks_data.filter(mask_inverted)
                if non_played_tracks.num_rows > 0:
                    for row in non_played_tracks.to_pylist():
                        Merge_data.append({"song_id": row["song_id"], "track_id": row["track_id"],
                                        "artist_name": row["artist_name"], "title": row["title"],
                                        "play_count": 0})

                if len(Merge_data) > 0:
                    table = pa.Table.from_pylist(
                        Merge_data,
                        schema=pa.schema([("song_id", pa.string()),
                                        ("track_id", pa.string()),
                                        ("artist_name", pa.string()),
                                        ("title", pa.string()),
                                        ("play_count", pa.int64())],)
                    )

                    if writer is None:
                        writer = pq.ParquetWriter(Merged_data_path, schema=table.schema)
                    writer.write_table(table)
                    Merge_data.clear()
                pbar.update(1)
    except BaseException as e:
        if writer is not None:
            writer.close()
        raise

    if writer is None:
        raise ValueError("No data found in file")
    writer.close()
    print(f"Song Data Merged successfully!!")

def Merge_Tracks_Genre(Data_folder):
    Merged_tracks_path = os.path.join(Data_folder, "Merged_tracks_data.parquet")
    if get_number_of_rows(Merged_tracks_path) >= 1000000:
        print(f"File Merged_tracks already converted to parquet!!")
        return
    if os.path.exists(Merged_tracks_path):
        os.remove(Merged_tracks_path)
        
    msd = ParquetFile(os.path.join(Data_folder, "msd_tagtraum_cd2.parquet")).read()
    Merged_songs = ParquetFile(os.path.join(Data_folder, "Merged_Song_data.parquet"))

    num_row_groups = Merged_songs.metadata.num_row_groups
    writer = None
    Merged_data = []
    try:
        with tqdm(total=int(num_row_groups), desc="Merging Genre Data") as pbar:
            for rg in range(num_row_groups):
                table = Merged_songs.read_row_group(rg)
                mask = pc.is_in(msd["track_id"], table["track_id"])
                filtered_data = msd.filter(mask)
                for row in filtered_data.to_pylist():
                    mask = pc.equal(table["track_id"], row["track_id"])
                    for merged_row in table.filter(mask).to_pylist():
                        Merged_data.append({"track_id": row["track_id"], "majority_genre": row["majority_genre"],
                                            "minority_genre": row["minority_genre"], "artist_name": merged_row["artist_name"],
                                            "title": merged_row["title"], "play_count": merged_row["play_count"],
                                            "song_id": merged_row["song_id"]})

                inverted_mask = pc.is_in(table["track_id"], msd["track_id"])
                inverted_mask = pc.invert(inverted_mask)
                non_merged_data = table.filter(inverted_mask)
                for row in non_merged_data.to_pylist():
                    Merged_data.append({"track_id": row["track_id"], "majority_genre": "",
                                        "minority_genre": "", "artist_name": row["artist_name"],
                                        "title": row["title"], "play_count": row["play_count"],
                                        "song_id": row["song_id"]})

                if len(Merged_data) > 0:
                    table = pa.Table.from_pylist(
                        Merged_data,
                        schema=pa.schema([("track_id", pa.string()),
                                        ("song_id", pa.string()),
                                        ("title", pa.string()),
                                        ("artist_name", pa.string()),
                                        ("majority_genre", pa.string()),
                                        ("minority_genre", pa.string()),
                                        ("play_count", pa.int64())
                                        ],)
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(Merged_tracks_path, schema=table.schema)
                    writer.write_table(table)
                    Merged_data.clear()

                pbar.update(1)
    except BaseException as e:
        if writer is not None:
            writer.close()
        raise
    if writer is None:
        raise ValueError("No data found in file")
    writer.close()
    print(f"Tracks Data Merged successfully!!")

def Sort_data(Data_folder):
    Merged_data_path = os.path.join(Data_folder, "Merged_tracks_data.parquet")
    parquet_data = ParquetFile(Merged_data_path)
    metadata = parquet_data.metadata.metadata
    if metadata.get(b"sorted", b"N").decode() == "Y":
        print("Data already sorted!!")
        return
    table = parquet_data.read()
    table = table.sort_by([("play_count", "descending")])
    new_metadata = dict(metadata)
    new_metadata[b"sorted"] = b"Y"
    table = table.replace_schema_metadata(new_metadata)
    writer = None
    try:
        writer = pq.ParquetWriter(Merged_data_path, schema=table.schema)
        writer.write_table(table)
    except Exception as e:
        if writer is not None:
            writer.close()
        raise Exception(f"Error in writing data {e}")

    print("Data Sorted successfully!!")

    if os.path.exists(os.path.join(Data_folder, "play_count.parquet")):
        os.remove(os.path.join(Data_folder, "play_count.parquet"))
        print("play_count.parquet removed!!")
    if os.path.exists(os.path.join(Data_folder, "Merged_Song_data.parquet")):
        os.remove(os.path.join(Data_folder, "Merged_Song_data.parquet"))
        print("Merged_Song_data.parquet removed!!")

def Merge_All_Data(Data_folder):
    Merged_tracks_path = os.path.join(Data_folder, "Merged_tracks_data.parquet")
    if get_number_of_rows(Merged_tracks_path) >= 1000000:
        print(f"File Merged_tracks already converted to parquet!!")
        Sort_data(Data_folder)
        return
    Merge_Song_Data(Data_folder)
    Merge_Tracks_Genre(Data_folder)
    Sort_data(Data_folder)