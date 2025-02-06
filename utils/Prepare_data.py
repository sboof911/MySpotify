from tqdm import tqdm
import pyarrow.compute as pc
from pyarrow.parquet import ParquetFile
from pyarrow import schema



DATA_FOLDER = "PreproccessedData"

def play_count_prepare(trainTriplets : ParquetFile, path : str):
    num_row_groups = trainTriplets.metadata.num_row_groups
    save_data_folder = f"{path}/{DATA_FOLDER}"
    save_data_path = f"{save_data_folder}/play_count.parquet"
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
                  for song_id, play_count in sorted(data_count.items(), key=lambda x:x[1], reverse=True)]

    return data_count