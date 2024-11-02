import os
import json
import time
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from transformers import BertTokenizer, BertForMaskedLM

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')


def get_contextual_synonym(word, context, bert_model, tokenizer):
    """获取上下文相关的同义词"""
    tokens = tokenizer.tokenize(context)
    if word not in tokens:
        return word  # 如果词不在上下文中，则返回原词

    word_idx = tokens.index(word)
    masked_tokens = tokens.copy()
    masked_tokens[word_idx] = tokenizer.mask_token

    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    inputs = tokenizer(masked_text, return_tensors='pt')

    with torch.no_grad():
        outputs = bert_model(**inputs)
        predictions = outputs.logits

    predicted_ids = torch.topk(predictions[0, word_idx], k=10).indices.tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    predicted_tokens = [token for token in predicted_tokens if
                        token != tokenizer.convert_tokens_to_string(tokenizer.tokenize(word))]

    if not predicted_tokens:
        return word

    synonym = predicted_tokens[0]
    return tokenizer.convert_tokens_to_string([synonym])


def get_wordnet_pos(treebank_tag):
    """将 NLTK 的词性标记转换为 WordNet 的词性标记"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def random_synonym_replacement(text, replacement_prob, bert_model, tokenizer):
    """上下文相关同义词替换，仅针对名词、形容词、副词和数词"""
    words = text.split()
    tagged_words = pos_tag(words)
    new_words = []

    for word, tag in tagged_words:
        pos = get_wordnet_pos(tag)
        if pos and random.random() < replacement_prob:
            synonyms = set()
            for syn in wordnet.synsets(word, pos=pos):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
            if word in synonyms:
                synonyms.remove(word)  # 移除自身
            if synonyms:
                new_word = random.choice(list(synonyms))
                context = ' '.join(words)
                new_word = get_contextual_synonym(new_word, context, bert_model, tokenizer)
                new_words.append(new_word)
            else:
                new_words.append(word)  # 如果没有同义词
        else:
            new_words.append(word)

    return ' '.join(new_words)


def process_conversation(text, replacement_prob, bert_model, tokenizer):
    """处理对话文本，仅替换说话人发言部分"""
    parts = text.split(':', 1)  # 按照第一个冒号分割
    if len(parts) == 2:
        speaker = parts[0]
        sentence = parts[1].strip()
        new_sentence = random_synonym_replacement(sentence, replacement_prob, bert_model, tokenizer)
        return f"{speaker}: {new_sentence}"
    return text  # 如果文本格式不匹配，直接返回原文本


def random_swap(dialogues):
    """随机交换对话中的两个句子"""
    if len(dialogues) < 2:
        return dialogues  # 如果对话不足两句，不进行交换
    idx1, idx2 = random.sample(range(len(dialogues)), 2)
    dialogues[idx1], dialogues[idx2] = dialogues[idx2], dialogues[idx1]
    return dialogues


def random_delete(dialogues, delete_prob):
    """随机删除对话中的某个句子"""
    if len(dialogues) <= 1:
        return dialogues  # 如果只有一个句子，不删除
    return [sentence for sentence in dialogues if random.random() > delete_prob]


def augment_dialogue(dialogues, bert_model, tokenizer, replacement_prob=0.2):
    """综合应用随机替换、随机交换和随机删除的增强策略"""
    # # 随机交换句子  US
    # if random.random() < swap_prob:
    #     dialogues = random_swap(dialogues)
    #
    # # 随机删除句子
    # dialogue = random_delete(dialogues, delete_prob)

    # 对每个句子应用随机同义词替换
    dialogue_augmentation = []
    for sentence in dialogues:
        if len(sentence) > 25:
            new_text = process_conversation(sentence, replacement_prob, bert_model, tokenizer)
            dialogue_augmentation.append(new_text)
        else:
            dialogue_augmentation.append(sentence)
    return dialogues


def process_sr(input_folder, output_file):
    # input_folder = './fenju/output-1p/dialogsum-1p'
    # output_file = './processed_output-US.jsonl'

    # 初始化模型
    model_path = "../model/sbert-base"
    sentence_model = SentenceTransformer(model_path)

    # 加载 BERT 模型和 tokenizer
    tokenizer = BertTokenizer.from_pretrained('../model/bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('../model/bert-base-uncased')

    # 打开文件用于追加
    with open(output_file, 'w', encoding='utf-8') as output_file:
        # 遍历输入文件夹中的所有 JSON 文件
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.json'):
                # 从JSON文件中读取数据
                with open(os.path.join(input_folder, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 获取对话列表和摘要
                dialogues = data['dialogue']
                summary = data['summary']

                # 计算摘要和对话句子的嵌入
                emb_summary = sentence_model.encode([summary], convert_to_tensor=True)
                emb_sentences = sentence_model.encode(dialogues, convert_to_tensor=True)
                # 检查形状
                if emb_summary.ndim != 2 or emb_sentences.ndim != 2:
                    print(f"Error: Embeddings are not 2D in {file_name}, skipping.")
                    continue
                # 计算余弦相似度
                similarities = cosine_similarity(emb_summary.cpu(), emb_sentences.cpu())

                # 找出相似度最高的句子
                max_idx = similarities.argmax()
                max_sentence = dialogues[max_idx]

                # 增强对话
                augmented_dialogue = augment_dialogue(dialogues, bert_model, tokenizer, replacement_prob=0.2)

                # 构建新的字典
                new_data = {
                    "dialogue": ' '.join(augmented_dialogue),
                    "summary": summary,
                }
                # 将新字典保存到jsonl文件中
                output_file.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        print("Processing completed.")
