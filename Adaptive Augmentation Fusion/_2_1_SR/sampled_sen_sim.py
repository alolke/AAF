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

# Get a synonym for a word based on its context.
def get_contextual_synonym(word, context, bert_model, tokenizer):
    tokens = tokenizer.tokenize(context)
    if word not in tokens:
        return word

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

# Convert NLTK POS tags to WordNet POS tags.
def get_wordnet_pos(treebank_tag):
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

# Replace words with synonyms based on context.
def random_synonym_replacement(text, replacement_prob, bert_model, tokenizer):
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
                synonyms.remove(word)
            if synonyms:
                new_word = random.choice(list(synonyms))
                context = ' '.join(words)
                new_word = get_contextual_synonym(new_word, context, bert_model, tokenizer)
                new_words.append(new_word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)

    return ' '.join(new_words)

# Apply replacement to the speaker's sentence in dialogue.
def process_conversation(text, replacement_prob, bert_model, tokenizer):
    parts = text.split(':', 1)
    if len(parts) == 2:
        speaker = parts[0]
        sentence = parts[1].strip()
        new_sentence = random_synonym_replacement(sentence, replacement_prob, bert_model, tokenizer)
        return f"{speaker}: {new_sentence}"
    return text

# Randomly swap two sentences in the dialogue.
def random_swap(dialogues):
    if len(dialogues) < 2:
        return dialogues
    idx1, idx2 = random.sample(range(len(dialogues)), 2)
    dialogues[idx1], dialogues[idx2] = dialogues[idx2], dialogues[idx1]
    return dialogues

# Randomly delete sentences from the dialogue.
def random_delete(dialogues, delete_prob):
    if len(dialogues) <= 1:
        return dialogues  # 如果只有一个句子，不删除
    return [sentence for sentence in dialogues if random.random() > delete_prob]

# Enhance dialogues with synonym replacement.
def augment_dialogue(dialogues, bert_model, tokenizer, replacement_prob=0.2):
    dialogue_augmentation = []
    for sentence in dialogues:
        if len(sentence) > 25:
            new_text = process_conversation(sentence, replacement_prob, bert_model, tokenizer)
            dialogue_augmentation.append(new_text)
        else:
            dialogue_augmentation.append(sentence)
    return dialogues

# Process dialogues with augmentation and save the output
def process_sr(input_folder, output_file):

    model_path = "../model/sbert-base"
    sentence_model = SentenceTransformer(model_path)

    tokenizer = BertTokenizer.from_pretrained('../model/bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('../model/bert-base-uncased')

    with open(output_file, 'w', encoding='utf-8') as output_file:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.json'):
                with open(os.path.join(input_folder, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dialogues = data['dialogue']
                summary = data['summary']

                # Calculate similarity between summary and sentences
                emb_summary = sentence_model.encode([summary], convert_to_tensor=True)
                emb_sentences = sentence_model.encode(dialogues, convert_to_tensor=True)

                if emb_summary.ndim != 2 or emb_sentences.ndim != 2:
                    print(f"Error: Embeddings are not 2D in {file_name}, skipping.")
                    continue
                # cos-sim
                similarities = cosine_similarity(emb_summary.cpu(), emb_sentences.cpu())
                max_idx = similarities.argmax()
                max_sentence = dialogues[max_idx]

                augmented_dialogue = augment_dialogue(dialogues, bert_model, tokenizer, replacement_prob=0.2)

                new_data = {
                    "dialogue": ' '.join(augmented_dialogue),
                    "summary": summary,
                }

                output_file.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        print("Processing completed.")
