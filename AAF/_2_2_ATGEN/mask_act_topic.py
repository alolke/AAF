import ast
import os
import json
import random
import re
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from safetensors.torch import load_file
import ast

# Load model
model_checkpoint_path = './act-gen-model-SAMSUM/checkpoint-624'  # model_path
safetensors_file = './act-gen-model-SAMSUM/checkpoint-624/model.safetensors'  # safetensors_path

# Load BART model architecture
model = BartForConditionalGeneration.from_pretrained(model_checkpoint_path,
                                                     ignore_mismatched_sizes=True)
# Load weights using the safetensors library
state_dict = load_file(safetensors_file)
# Load the weights into the model
model.load_state_dict(state_dict, strict=False)
# Load tokenizer
tokenizer = BartTokenizer.from_pretrained(model_checkpoint_path)

# Load each line from the topic_idx file
def load_topic_indices(topic_file):
    topic_indices_list = []
    with open(topic_file, 'r', encoding="utf-8") as f:
        for line in f:
            # Read each line as a list and convert it into an array
            topic_indices_list.append(ast.literal_eval(line.strip()))
    return topic_indices_list


def remove_consecutive_masks(sentence):
    # Replace consecutive <MASK> tokens with a single <MASK> using regex
    return re.sub(r'(<MASK>\s*)+', '<MASK> ', sentence)


# Process each file
def process_json_file(file_path, topic_indices):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

        src = data['src']
        events = data['event']
        summary = data['summary']
        # Divide sentences into different parts based on topic_indices
        topics = {}
        for idx, topic_id in enumerate(topic_indices):
            if topic_id not in topics:
                topics[topic_id] = []
            topics[topic_id].append(idx)

        masked_indices = []
        # Check the number of topic segments
        if len(topics) == 1:
            # If there is only one topic segment, randomly select one sentence to mask
            for idx, event in enumerate(events):
                if event != "":
                    masked_indices.append(idx)
                    break
        else:
            # If there are multiple topic segments, randomly select one and mask all sentences in that segment
            selected_topic = random.choice(list(topics.values()))
            masked_indices = selected_topic  # Directly use all sentence indexes in the topic section to read each line as a list and convert it into an array

        # Construct `src` and `gen`
        masked_src = []  # Sentences with masks
        mask_prompt = []  # Model generation prompts
        for i, sentence in enumerate(src):
            if i in masked_indices:
                person, _ = sentence.split(":", 1)
                masked_src.append('<MASK>')
                mask_prompt.append(f'{person}:{events[i]}')
            else:
                masked_src.append(sentence)

        # Generate `src` and `gen` texts
        masked_text = remove_consecutive_masks(" ".join(masked_src))
        prompt_text = ",".join(mask_prompt)

        prompt = f"{prompt_text}</s>{masked_text}"
        # Generate dialogue using the model
        inputs = tokenizer(prompt, return_tensors='pt')
        # Generate output
        output_sequences = model.generate(**inputs)
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        Augmentation_data = masked_text.replace('<MASK>', generated_text, 1)

        data = {
            "dialogue": Augmentation_data,
            "summary": summary,
        }

        return data


# Process all JSON files in a folder
def process_all_json_files(folder_path, topic_file):
    results = []
    topic_indices_list = load_topic_indices(topic_file)  # Load all topic indices
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]   # Get all JSON files

    for i, filename in enumerate(files):
        if i < len(topic_indices_list):  #  Ensure the number of files matches the topic indices
            file_path = os.path.join(folder_path, filename)
            topic_indices = topic_indices_list[i]  # Get corresponding topic indices
            result = process_json_file(file_path, topic_indices)
            results.append(result)

    return results


def act_top_gen(act_folfer, topic_idx_file, DA_output):
    output = process_all_json_files(act_folfer, topic_idx_file)
    # Save results to a new file
    with open(DA_output, 'w', encoding="utf-8") as out_file:
        for item in output:
            json.dump(item, out_file)
            out_file.write("\n")
