from transformers import BartForConditionalGeneration, BartTokenizer
import json
from datasets import load_metric  # pip install datasets==2.7.1
import numpy as np
metric = load_metric("./rouge.py")


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def process_json_files(datas, model, tokenizer):
    results = []
    for data in datas:
        input = data["dialogue"]
        original_summary = data["summary"]
        # generation summary
        input = tokenizer(input, return_tensors='pt')
        output_sequences = model.generate(**input)
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # ROUGE score
        rouge_results = metric.compute(predictions=[generated_text], references=[original_summary])
        rouge_score = rouge_results['rougeL'][0].fmeasure  # 选择 ROUGE-L F1 分数

        results.append((generated_text, rouge_score))

    return results


def save_list_to_specific_line(filename, list_to_save, line_num):

    with open(filename, 'r') as file:
        lines = file.readlines()
    if line_num >= len(lines):
        lines.extend(['\n'] * (line_num - len(lines) + 1))
    lines[line_num] = ','.join(map(str, list_to_save)) + '\n'
    with open(filename, 'w') as file:
        file.writelines(lines)


def load_list_from_line(filename, line_number):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if 0 <= line_number < len(lines):
            line = lines[line_number].strip()
            # 使用逗号分隔字符串，并将结果转换为numpy.float64类型的列表
            lst = [np.float64(x) for x in line.split(',')]
            return lst
        else:
            raise IndexError("Line number out of range")


def Rouge_compute(adjustable_para, trainset_file, model_checkpoint_path, line_num):
    datas = read_jsonl(trainset_file)

    # Compute ROUGE scores
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint_path)
    tokenizer = BartTokenizer.from_pretrained(model_checkpoint_path)

    # Compute ROUGE scores
    inference_output = process_json_files(datas, model, tokenizer)

    # Initialize ratio as 0.5:0.5
    percentage = 0.5

    # Extract ROUGE scores
    rouge_scores = [score for i, (summary, score) in enumerate(inference_output)]

    # Save, retrieve, and compare scores
    save_list_to_specific_line("./Trend_split/trainset_rouge.txt", rouge_scores, line_num)
    loading_rouge = load_list_from_line("./Trend_split/trainset_rouge.txt", line_num - 1)

    # Calculate the differences between the current and previous scores using list comprehension
    result_list = [a - b for a, b in zip(rouge_scores, loading_rouge)]

    # Combine indices with scores
    indexed_scores = [(i, score) for i, score in enumerate(result_list)]
    # Sort scores in descending order
    ranked_indices = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    # Compute split index
    split_index = int(len(ranked_indices) * (percentage + adjustable_para))

    # Split into top and bottom indices
    top_indices = [idx for idx, _ in ranked_indices[:split_index]]
    bottom_indices = [idx for idx, _ in ranked_indices[split_index:]]

    # Save the top 50% and bottom 50% data into separate JSONL files
    def save_to_jsonl(data_indices, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx in data_indices:
                json.dump(datas[idx], f, ensure_ascii=False)
                f.write('\n')

    # save
    save_to_jsonl(top_indices, './data/augmentation-tmp/Good_data.jsonl')
    save_to_jsonl(bottom_indices, './data/augmentation-tmp/Bad_data.jsonl')


# Rouge_compute(0.5 ,"../data/dialogsum-1p-2.jsonl", "../../model/bart-base", 0)
