import json
import random


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def random_sample(data, sample_size):
    return random.sample(data, sample_size)


def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    file_path = 'dialogsum.train.jsonl'  
    output_file_path = 'dialogsum-1p-3.jsonl'
    sample_size = 125

    # Read_JSONL files
    data = read_jsonl_file(file_path)

    if len(data) < sample_size:
        print(f"Warning: The number of data in the file is less than {sample_2}, and only all data can be returned.")
        sample_size = len(data)

        # Randomly select samples
    sample = random_sample(data, sample_size)

    write_jsonl_file(output_file_path, sample)


if __name__ == "__main__":
    main()
