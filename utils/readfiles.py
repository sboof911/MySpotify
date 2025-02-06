import os, json
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
from tqdm import tqdm

DATA_FOLDER = "parquet_files"
def get_file_path(path : str, filename : str) -> str:
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        raise ValueError(f"{filename} not found in path")
    return file_path

def get_zip_file(path : str, filename : str):
    file_path = get_file_path(path, filename)
    if not file_path.endswith(".zip"):
        raise ValueError(f"{filename} is not a zip file")
    file_txt = file_path.replace(".zip", "")
    if os.path.exists(file_txt):
        return file_txt
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        print(f"Extracting {filename}...")
        zip_ref.extractall(path)
    return file_txt

def check_data_folder(path : str):
    parquet_files_path = os.path.join(path, DATA_FOLDER)
    if not os.path.exists(parquet_files_path):
        os.makedirs(parquet_files_path, exist_ok=True)

def count_lines(parquet_save_path : str) -> int:
    try:
        metadata = pq.read_metadata(parquet_save_path)
        return metadata.num_rows
    except Exception as e:
        with open(parquet_save_path, "w") as file:
            file.write("")
        return 0

def Convert_To_Parquet(path : str, buffer_size):
    check_data_folder(path)
    prepare_msd(path, buffer_size)
    prepare_unique_tracks(path, buffer_size)
    prepare_mxm_dataset_train(path, buffer_size)
    prepare_train_triplets(path, buffer_size)

def check_file_existant(parquet_save_path, numOfLines):
    if os.path.exists(parquet_save_path):
        if count_lines(parquet_save_path) >= numOfLines:
            parquet_save_path[parquet_save_path.rfind("/")+1:-8]
            print(f"{parquet_save_path[parquet_save_path.rfind("/")+1:-8]} file already converted to parquet")
            return True
    return False

def get_data_dict(columns, data_list):
    data_dict = {}
    for i, (col_name, col_type) in enumerate(columns):
        if i < len(data_list):
            data_dict[col_name] = data_list[i] if col_type == pa.string() else int(data_list[i])
        else:
            if col_type == pa.string():
                data_dict[col_name] = ""
            else:
                data_dict[col_name] = 0
    return data_dict

def Convert_file_To_Parquet(path : str, filename : str, numOfLines,
                        columns, sep, buffer_size):
    parquet_save_path = f"{path}/{DATA_FOLDER}/{filename[:-4]}.parquet"
    file_path = get_file_path(path, filename)
    if check_file_existant(parquet_save_path, numOfLines):
        return
    print(f'Reading {filename[:-4]} file...')
    buffer = []
    batch_counter = 0
    writer = None
    pa_columns=pa.schema(columns)

    with open(file_path, "r", errors="replace") as infile:
        num_iterations = sum([1 for _ in infile]) // buffer_size
        infile.seek(0)
        with tqdm(total=num_iterations, desc=f"Converting {filename}") as pbar:
            for line in infile:
                line = line.replace("�", " ")
                if line.startswith('#'):
                    continue
                line = line.replace('\n', '')
                data_list = line.split(sep) if sep else line.split()
                buffer.append(get_data_dict(columns, data_list))

                if len(buffer) >= buffer_size:
                    table = pa.Table.from_pylist(
                        buffer,
                        schema=pa_columns
                    )
                    if batch_counter == 0:
                        writer = pq.ParquetWriter(parquet_save_path, schema=table.schema)

                    writer.write_table(table)
                    buffer.clear()
                    batch_counter += 1
                pbar.update(1)

    if buffer:
        table = pa.Table.from_pylist(
            buffer,
            schema=pa_columns
        )
        writer.write_table(table)
        buffer.clear()

    if writer is None:
        raise ValueError("No data found in file")

    if 'writer' in locals():
        writer.close()

    print(f"{filename[:-4]} file converted to parquet")

def prepare_msd(path : str, buffer_size):
    kwargs = {
        "path": path,
        "filename": "p02_msd_tagtraum_cd2.cls",
        "numOfLines": 280831,
        "columns": [("track_id", pa.string()),
                    ("majority_genre", pa.string()),
                    ("minority_genre", pa.string())],
        "sep": "\t",
        "buffer_size":buffer_size
    }
    Convert_file_To_Parquet(**kwargs)

def prepare_unique_tracks(path : str, buffer_size):
    kwargs = {
        "path": path,
        "filename": "p02_unique_tracks.txt",
        "numOfLines": 1000000,
        "columns": [
            ("track_id", pa.string()),
            ("song_id", pa.string()),
            ("artist_name", pa.string()),
            ("title", pa.string())
        ],
        "sep": "<SEP>",
        "buffer_size":buffer_size
    }
    Convert_file_To_Parquet(**kwargs)

def prepare_train_triplets(path : str, buffer_size):
    kwargs = {
        "path": path,
        "filename": "train_triplets.txt",
        "numOfLines": 48373586,
        "columns": [
            ("user_id", pa.string()),
            ("song_id", pa.string()),
            ("play_count", pa.int64())
        ],
        "sep": "\t",
        "buffer_size":buffer_size
    }
    Convert_file_To_Parquet(**kwargs)

def prepare_mxm_dataset_train(path : str, buffer_size):
    json_words_path = f"{path}/{DATA_FOLDER}/words.json"
    filename = "mxm_dataset_train.txt"
    parquet_save_path = f"{path}/{DATA_FOLDER}/{filename[:-4]}.parquet"
    if check_file_existant(parquet_save_path, 210519):
        return

    print(f'Reading {filename[:-4]} file...')
    file_path = get_file_path(path, filename)

    tmp_file_path = parquet_save_path.replace(".parquet", "_tmp.txt")
    if not check_file_existant(tmp_file_path, 1):
        print("Creating a tmp txt file...")
        with open(tmp_file_path, "w") as outfile:
            with open(file_path, "r", encoding="utf_8", errors="replace") as infile:
                for line in infile:
                    if line.startswith("#"):
                        continue
                    if line.startswith("%"):
                        line = line[1:]
                        words = line.strip().split(",")
                        words_data = {"words": words}
                        with open(json_words_path, "w") as json_file:
                            json.dump(words_data, json_file)
                        continue
                    outfile.write(line)

    words = json.load(open(json_words_path))["words"]
    print("Processing Data...")

    buffer = []
    batch_counter = 0
    word_dict = {word: 0 for word in words}
    writer = None
    columns=pa.schema([
            ("track_id", pa.string()), 
            ("mxm_track_id", pa.int64())
        ] + [(word, pa.int64()) for word in words])
    with open(tmp_file_path, "r", encoding="utf_8") as infile:
        num_iterations = sum([1 for _ in infile]) // buffer_size
        infile.seek(0)
        with tqdm(total=num_iterations, desc=f"Converting {filename}") as pbar:
            for line in infile:
                word_dict_tmp = word_dict
                data = line.split(",")
                word_dict_tmp = {words[int(word_id) - 1]: int(count) for word_id, count in (item.strip().split(":") for item in data[2:])}
                word_dict_tmp = {word: word_dict_tmp.get(word, 0) for word in words}
                buffer.append({"track_id": data[0], "mxm_track_id": int(data[1]), **word_dict_tmp})

                if len(buffer) >= buffer_size:
                    table = pa.Table.from_pylist(
                        buffer, 
                        schema=columns
                    )
                    if batch_counter == 0:
                        writer = pq.ParquetWriter(parquet_save_path, schema=table.schema)

                    writer.write_table(table)
                    buffer.clear()
                    batch_counter += 1
                pbar.update(1)

    print("Saving Dataframe to parquet...")
    if buffer:
        table = pa.Table.from_pylist(
            buffer,
            schema=columns
        )
        writer.write_table(table)
        buffer.clear()

    if 'writer' in locals():
        writer.close()
    os.remove(tmp_file_path)
    print(f"Data successfully saved to {parquet_save_path}")