import json
import random


def read_jsonl_file(file_path):
    """
    读取JSONL文件并返回所有行的列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def random_sample(data, sample_size):
    """
    从数据中随机抽取指定数量的样本。
    """
    return random.sample(data, sample_size)


def write_jsonl_file(file_path, data):
    """

    将数据写入JSONL文件。

    """

    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    file_path = 'dialogsum.train.jsonl'  # 替换为你的JSONL文件路径
    output_file_path = 'dialogsum-1p-3.jsonl'
    sample_size = 125

    # 读取JSONL文件
    data = read_jsonl_file(file_path)

    # 检查数据是否足够
    if len(data) < sample_size:
        print(f"警告：文件中的数据少于{sample_size}个，只能返回全部数据。")
        sample_size = len(data)

        # 随机抽取样本
    sample = random_sample(data, sample_size)

    write_jsonl_file(output_file_path, sample)


if __name__ == "__main__":
    main()
