import warnings

warnings.filterwarnings('ignore')

import json
from datasets import load_metric, Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    TrainerCallback, set_seed
from transformers import AutoTokenizer
import torch
import torch.backends.cudnn as cudnn
import os
import nltk
import numpy as np
import random
import logging
import shutil

# Fixed random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # using multiple GPUs
cudnn.deterministic = True
cudnn.benchmark = False
set_seed(seed)


from Regulator import Regulator
from trainset_inference import trainset_inference
from _1_fenju import fenju
from _2_1_SR import process_sr
from _2_2_topic_annotation import topic_annotation
from _2_2_action_extraction import act_ext
from _2_2_ATGEN import act_top_gen
from Trend_split import Rouge_compute

# Suppress NLTK download messages
logging.getLogger('nltk').setLevel(logging.ERROR)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

TEST_SUMMARY_ID = 1


# ---------- Utility Function: Clear Directory ----------
# Deletes all contents of a directory and recreates it as an empty folder.
def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


# ---------- Data Transformation Functions ----------
# Converts a single DialogSum data file to HuggingFace dataset format.
def transform_single_dialogsumm_file(file):
    data = open(file, "r", encoding='utf-8').readlines()
    result = {"summary": [], "dialogue": []}
    for i in data:
        d = json.loads(i)
        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])
    return Dataset.from_dict(result)


# Converts test files with specific formats to HuggingFace dataset
def transform_test_file(file):
    data = open(file, "r").readlines()
    result = {"fname": [], "summary%d" % TEST_SUMMARY_ID: [], "dialogue": []}
    for i in data:
        d = json.loads(i)
        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])

    result["summary"] = result["summary%d" % TEST_SUMMARY_ID]
    return Dataset.from_dict(result)

# Combines train, validation, and test sets into a HuggingFace dataset dictionary.
def transform_dialogsumm_to_huggingface_dataset(train, validation, test):
    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_test_file(test)
    return DatasetDict({"train": train, "validation": validation, "test": test})


# ---------- Callback for Logging and Validation ----------
# Custom callback for tracking evaluation results during training.
class LoggingAndComparisonCallback(TrainerCallback):
    def __init__(self):
        self.eval_results = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.eval_results.append(metrics)

    def get_eval_results(self):
        return self.eval_results


# ---------- Model Loading ----------
# Finds and loads the single saved model checkpoint from a directory.
def loading_model(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    checkpoint_files = [f for f in files if f.startswith('checkpoint')]

    if len(checkpoint_files) == 1:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])

    return checkpoint_path


# ---------- Sampling Utility ----------
# Samples a percentage of data from a given JSONL file.
def sample_data(file_path, sample_rate=0.1):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    sample_size = int(len(data) * sample_rate)
    return random.sample(data, sample_size)


# ---------- Fusion Augmentation Training ----------
# Implements the core training pipeline with adaptive data augmentation.
def Fusion_augmentaton_Training(num, pre_adjustable_para, stop_threshold):
    # Set initial model checkpoint and training data
    if num == 1:
        model_checkpoint = "../model/bart-base"
        trainset = "./data/dialogsum-1p-2.jsonl"
    else:
        model_checkpoint = loading_model(r"./fusion-model")
        trainset = f"./data/AugmentationData/dialogsum-1p-DA-{num - 1}.jsonl"
    # Load and transform datasets
    raw_datasets = transform_dialogsumm_to_huggingface_dataset(trainset,
                                                               "./data/dialogsum.dev.jsonl",
                                                               "./data/dialogsum.test.jsonl")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

    # Preprocess data
    def preprocess_function(examples):
        inputs = [doc for doc in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Load evaluation metric (ROUGE)
    metric = load_metric("./rouge.py")

    # Define function for computing evaluation metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    max_input_length = 256
    max_target_length = 128

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    batch_size = 32
    num_train_epochs = 2

    args = Seq2SeqTrainingArguments(
        output_dir=f"./fusion-model",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
        save_total_limit=1,
        metric_for_best_model="eval_rouge1",
        load_best_model_at_end=True,  
        greater_is_better=True,
        seed=100,
        generation_max_length=max_target_length,
    )
    # warmup_steps=200,
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize trainer with callback
    comparison_callback = LoggingAndComparisonCallback()

    # Train the model and manage evaluation results
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[comparison_callback]
    )

    if os.path.exists(".r/fusion-model"):
        shutil.rmtree(".r/fusion-model")
        os.makedirs(".r/fusion-model")
    trainer.train()

    # Validate before testing, enhancing steps
    # Obtain evaluation results
    eval_results = comparison_callback.get_eval_results()
    adjustable_para, Stop_symbol, stop_threshold = Regulator(eval_results, pre_adjustable_para, stop_threshold)

    # Rank each data point by comparing the current validation ROUGE score with the difference from the previous step.
    # If the ROUGE score increases, the data is considered good; otherwise, it is considered bad.
    Rouge_compute(adjustable_para, "./data/dialogsum-1p-2.jsonl",
                  model_checkpoint_path=loading_model(r"./fusion-model"), line_num=num)

    good_data_file = './data/augmentation-tmp/Good_data.jsonl'
    bad_data_file = './data/augmentation-tmp/Bad_data.jsonl'

    # Before enhancement, preprocess the data into a specific structure with sentence segmentation
    # Segment the data folder, required for subsequent action extraction and topic segmentation
    # Clear the good_data folder
    if os.path.exists(r"./data/Bad_data"):
        shutil.rmtree(r"./data/Bad_data")
        os.makedirs(r"./data/Bad_data")
    fenju(bad_data_file, f"./data")
    BD_Segmented_folder = f"./data/Bad_data"

    # Data augmentation files
    DA_SR = "./data/augmentation-tmp/Augmentation_data-simple.jsonl"
    DA_GEN = "./data/augmentation-tmp/Augmentation_data-diversity.jsonl"
    """-----------MPA Augmentation--------------"""
    # Perform MPA augmentation on good data
    process_sr(BD_Segmented_folder, DA_SR)

    """----------SRA Augmentation--------------"""
    # Perform SRA augmentation on bad data
    topic_annotation_file = "./_2_2_topic_annotation/topic_idx_1p.txt" # Define the topic segmentation file
    act_folder = './_2_2_action_extraction/act_1p'  # Define the action extraction file
    topic_annotation(good_data_file, topic_annotation_file)  # Topic segmentation
    act_ext(good_data_file, act_folder)  # Action extraction
    # Generate augmented data
    act_top_gen(act_folder, topic_annotation_file, DA_GEN)

    """
         Full Augmentation
    """

    # combined_data = []
    # with open(DA_SR, 'r', encoding='utf-8') as f1:
    #     for line in f1:
    #         combined_data.append(json.loads(line))
    # with open(DA_GEN, 'r', encoding='utf-8') as f2:
    #     for line in f2:
    #         combined_data.append(json.loads(line))
    # # with open("./data/dialogsum-1p-2.jsonl", 'r', encoding='utf-8') as f3:
    # with open(trainset, 'r', encoding='utf-8') as f3:
    #     for line in f3:
    #         combined_data.append(json.loads(line))

    """
        Partial Augmentation
    """
    combined_data = []
    # Sample 10% of data from DA_SR
    DA_SR_sample = sample_data(DA_SR, sample_rate=0.5)
    combined_data.extend(DA_SR_sample)
    # Sample 10% of data from DA_GEN
    DA_GEN_sample = sample_data(DA_GEN, sample_rate=0.5)
    combined_data.extend(DA_GEN_sample)
    # Read all data from the trainset file
    with open("./data/dialogsum-1p-2.jsonl", 'r', encoding='utf-8') as f3:
        for line in f3:
            combined_data.append(json.loads(line))

    # Shuffle the combined data
    random.shuffle(combined_data)
    # Save to a new file
    output_file = f'./data/AugmentationData/dialogsum-1p-DA-{num}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in combined_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # test

    out = trainer.predict(tokenized_datasets["test"], num_beams=5)

    predictions, labels, metric = out
    print(metric)
    with open(f"./metrics.txt", "a", encoding='utf-8') as f:
        if isinstance(metric, dict):
            f.write(str(metric) + "\n")
        else:
            f.write(str(metric) + "\n")

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after e ach sentence
    decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    # output summaries on test set
    with open(f"./output_text/1p-{num}.txt", "w", encoding='utf-8') as f:
        for i in decoded_preds:
            f.write(i.replace("\n", "") + "\n")

    return adjustable_para, Stop_symbol, stop_threshold
