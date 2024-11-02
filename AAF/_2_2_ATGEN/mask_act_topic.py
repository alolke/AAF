import ast
import os
import json
import random
import re
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from safetensors.torch import load_file
import ast

# 加载模型
# # 模型和配置文件的路径
# model_checkpoint_path = '../act-gen-model-SAMSUM/checkpoint-624'  # 模型文件夹路径
# # 加载 BART 模型架构
# model = BartForConditionalGeneration.from_pretrained(model_checkpoint_path,
#                                                      ignore_mismatched_sizes=True)
# # 加载分词器
# tokenizer = BartTokenizer.from_pretrained(model_checkpoint_path, use_fast=False)

model_checkpoint_path = './act-gen-model-SAMSUM/checkpoint-624'  # 模型文件夹路径
safetensors_file = './act-gen-model-SAMSUM/checkpoint-624/model.safetensors'  # 模型的safetensors文件路径

# 加载 BART 模型架构
model = BartForConditionalGeneration.from_pretrained(model_checkpoint_path,
                                                     ignore_mismatched_sizes=True)
# 使用safetensors库加载safetensors格式的权重
state_dict = load_file(safetensors_file)
# 将加载的权重加载到模型中
model.load_state_dict(state_dict, strict=False)
# 加载分词器
tokenizer = BartTokenizer.from_pretrained(model_checkpoint_path)

# 处理数据
# # 文件夹路径
# folder_path = '../action_extraction/output-1p'
# topic_idx_file = "../../data/topic_idx_1p.txt"


# 读取topic_idx文件的每一行
def load_topic_indices(topic_file):
    topic_indices_list = []
    with open(topic_file, 'r', encoding="utf-8") as f:
        for line in f:
            # 每一行读取为一个列表，转化为数组
            topic_indices_list.append(ast.literal_eval(line.strip()))
    return topic_indices_list


def remove_consecutive_masks(sentence):
    # 使用正则表达式替换连续的 <MASK> 为单个 <MASK>
    return re.sub(r'(<MASK>\s*)+', '<MASK> ', sentence)


# 处理每个文件
def process_json_file(file_path, topic_indices):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

        src = data['src']
        events = data['event']
        summary = data['summary']
        # 通过 topic_indices 将句子分为不同的部分
        topics = {}
        for idx, topic_id in enumerate(topic_indices):
            if topic_id not in topics:
                topics[topic_id] = []
            topics[topic_id].append(idx)

        masked_indices = []
        # 检查主题部分数量
        if len(topics) == 1:
            # 只有一个主题部分，随机选择一个句子进行掩蔽
            for idx, event in enumerate(events):
                if event != "":
                    masked_indices.append(idx)
                    break
        else:
            # 有多个主题部分，随机选择一个主题部分并掩蔽该部分的所有句子
            selected_topic = random.choice(list(topics.values()))
            masked_indices = selected_topic  # 直接使用该主题部分的所有句子索引

        # 构建src和gen
        masked_src = []  # 带mask标注的句子
        mask_prompt = []  # 模型生成提示
        for i, sentence in enumerate(src):
            if i in masked_indices:
                person, _ = sentence.split(":", 1)
                masked_src.append('<MASK>')
                mask_prompt.append(f'{person}:{events[i]}')
            else:
                masked_src.append(sentence)

        # 生成src文本和gen文本
        masked_text = remove_consecutive_masks(" ".join(masked_src))
        prompt_text = ",".join(mask_prompt)

        prompt = f"{prompt_text}</s>{masked_text}"
        # 模型生成对话
        inputs = tokenizer(prompt, return_tensors='pt')
        # 生成输出
        output_sequences = model.generate(**inputs)
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        Augmentation_data = masked_text.replace('<MASK>', generated_text, 1)

        data = {
            "dialogue": Augmentation_data,
            "summary": summary,
        }

        return data


# 遍历文件夹中的所有JSON文件
def process_all_json_files(folder_path, topic_file):
    results = []
    topic_indices_list = load_topic_indices(topic_file)  # 加载所有topic索引
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]  # 获取所有json文件

    for i, filename in enumerate(files):
        if i < len(topic_indices_list):  # 确保文件数量与topic索引数量一致
            file_path = os.path.join(folder_path, filename)
            topic_indices = topic_indices_list[i]  # 获取对应的topic索引
            result = process_json_file(file_path, topic_indices)
            results.append(result)

    return results


def act_top_gen(act_folfer, topic_idx_file, DA_output):
    # 输出处理后的结果
    output = process_all_json_files(act_folfer, topic_idx_file)
    # 保存到新文件
    with open(DA_output, 'w', encoding="utf-8") as out_file:
        for item in output:
            json.dump(item, out_file)
            out_file.write("\n")
