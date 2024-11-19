from sentence_transformers import SentenceTransformer
import torch
import jsonlines
from .C99 import C99
import os
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

local_model_path = '../model/bert-base-nli-stsb-mean-tokens/'
embedder = SentenceTransformer(local_model_path)


def encode_conversation(input_folder):
    data = []
    # load data
    with open(input_folder, encoding='utf8') as json_file:
        data_ = jsonlines.Reader(json_file)
        for obj in data_:
            data.append(obj)

    sent = []
    for i in range(len(data)):
        if len(data[i]['dialogue'].split('\r\n')) > 1:
            sentences = data[i]['dialogue'].split('\r\n')
        else:
            sentences = data[i]['dialogue'].split('\n')
        sent.append(sentences)

    embeddings = []
    with torch.no_grad():
        # generation embedding
        for i in tqdm(range(len(sent))):
            embedding = embedder.encode(sent[i])
            embeddings.append(embedding)

    return embeddings


def encode_convs(output_file, data):
    model = C99(window=4, std_coeff=1)
    sent_label = []

    # Segmented processing
    for i in range(len(data)):
        boundary = model.segment(data[i])
        temp_labels = []
        l = 0
        for j in range(len(boundary)):
            if boundary[j] == 1:
                l += 1
            temp_labels.append(l)
        sent_label.append(temp_labels)

        with open(output_file, "w") as Data:
            for item in sent_label:
                Data.write(str(item) + '\n')

        Data.close()

    return sent_label


def topic_annotation(input_folder, output_file):
    embeddings = encode_conversation(input_folder)
    encode_convs(output_file, data=embeddings)
