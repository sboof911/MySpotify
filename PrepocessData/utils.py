import os, json

import pyarrow.parquet as pq
from pyarrow import string, int64, schema, Table

from tqdm import tqdm

def check_file_existance(file_path):
    if not os.path.exists(file_path):
        return False
    return True

def get_number_of_rows(parquet_path):
    try:
        metadata = pq.read_metadata(parquet_path)
        return metadata.num_rows
    except Exception as e:
        return 0

def get_data_dict(columns, data_list):
    data_dict = {}
    for i, (col_name, col_type) in enumerate(columns):
        if i < len(data_list):
            data_dict[col_name] = data_list[i] if col_type == string() else int(data_list[i])
        else:
            if col_type == string():
                data_dict[col_name] = ""
            else:
                data_dict[col_name] = 0
    return data_dict

def Convert_file_To_Parquet(file_path, output_dir, numOfLines, columns, sep, buffer_size=10000):
    if not check_file_existance(file_path):
        raise ValueError(f"File {file_path} not found")

    filename = os.path.basename(file_path)[:-4]
    output_path = os.path.join(output_dir, filename + ".parquet")

    if get_number_of_rows(output_path) >= numOfLines:
        print(f"File {filename} already converted to parquet!!")
        return
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f'Reading {filename} file...')
    buffer = []
    writer = None
    pa_columns=schema(columns)

    with open(file_path, "r", errors="replace") as infile:
        try:
            with tqdm(total=numOfLines, desc=f"Converting {filename}") as pbar:
                for line in infile:
                    line = line.replace("�", " ")
                    if line.startswith('#'):
                        continue
                    line = line.replace('\n', '')
                    data_list = line.split(sep) if sep else line.split()
                    buffer.append(get_data_dict(columns, data_list))

                    if len(buffer) >= buffer_size:
                        table = Table.from_pylist(
                            buffer,
                            schema=pa_columns
                        )
                        if writer is None:
                            writer = pq.ParquetWriter(output_path, schema=table.schema)

                        writer.write_table(table)
                        buffer.clear()
                    pbar.update(1)
        except BaseException as e:
            if writer is not None:
                writer.close()
            raise

    if buffer:
        table = Table.from_pylist(
            buffer,
            schema=pa_columns
        )
        try:
            writer.write_table(table)
        except Exception as e:
            if writer is not None:
                writer.close()
            raise Exception(f"Error in writing data {e}")
        buffer.clear()

    if writer is None:
        raise ValueError("No data found in file")

    writer.close()

    print(f"{filename} file converted to parquet")


############################################################################################################

def create_tmp_mxm(file_path, output_path, json_words_path, buffer_size=10000):
    tmp_file_path = output_path.replace(".parquet", "_tmp.txt")
    print("Creating a tmp txt file...")
    with open(file_path, "r", encoding="utf_8", errors="replace") as infile:
        with open(tmp_file_path, "w") as outfile:
            data = []
            while True:
                lines = infile.readlines(buffer_size)
                if not lines:
                    break
                for line in lines:
                    if line.startswith("#"):
                        continue
                    if line.startswith("%"):
                        line = line[1:]
                        words = line.strip().split(",")
                        words_data = {"words": words}
                        with open(json_words_path, "w") as json_file:
                            json.dump(words_data, json_file)
                        continue
                    data.append(line)
                outfile.write("".join(data))
                data.clear()

    return tmp_file_path

def prepare_mxm_dataset_train(file_path, output_dir, numOfLines, buffer_size=10000):
    json_words_path = os.path.join(output_dir, "words.json")
    if not check_file_existance(file_path):
        raise ValueError(f"File {file_path} not found")

    filename = os.path.basename(file_path)[:-4]
    output_path = os.path.join(output_dir, filename + ".parquet")

    if get_number_of_rows(output_path) >= numOfLines:
        print(f"File {filename} already converted to parquet!!")
        return
    if os.path.exists(output_path):
        os.remove(output_path)

    tmp_file_path = create_tmp_mxm(file_path, output_path, json_words_path, buffer_size)

    print(f'Reading {filename} file...')
    words = json.load(open(json_words_path))["words"]

    print("Processing Data...")
    buffer = []
    word_dict = {word: 0 for word in words}
    writer = None
    columns=schema([
            ("track_id", string()), 
            ("mxm_track_id", int64())
        ] + [(word, int64()) for word in words])
    with open(tmp_file_path, "r", encoding="utf_8") as infile:
        try:
            with tqdm(total=numOfLines, desc=f"Converting {filename}") as pbar:
                for line in infile:
                    word_dict_tmp = word_dict
                    data = line.split(",")
                    word_dict_tmp = {words[int(word_id) - 1]: int(count) for word_id, count in (item.strip().split(":") for item in data[2:])}
                    word_dict_tmp = {word: word_dict_tmp.get(word, 0) for word in words}
                    buffer.append({"track_id": data[0], "mxm_track_id": int(data[1]), **word_dict_tmp})

                    if len(buffer) >= buffer_size:
                        table = Table.from_pylist(
                            buffer, 
                            schema=columns
                        )
                        if writer is None:
                            writer = pq.ParquetWriter(output_path, schema=table.schema)

                        writer.write_table(table)
                        buffer.clear()
                    pbar.update(1)
        except BaseException as e:
            if writer is not None:
                writer.close()
            raise

    print("Saving Dataframe to parquet...")
    if buffer:
        table = Table.from_pylist(
            buffer,
            schema=columns
        )
        try:
            writer.write_table(table)
        except Exception as e:
            if writer is not None:
                writer.close()
            raise Exception(f"Error in writing data {e}")
        buffer.clear()

    if writer is None:
        raise ValueError("No data found in file")
    writer.close()
    os.remove(tmp_file_path)

    print(f"{filename} file converted to parquet")