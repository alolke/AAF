from nltk import sent_tokenize
import nltk
from os.path import join
import os
import json
from time import time
import tqdm
import jsonlines

nltk.download('punkt')

def fenju(input_file, output_path):
    # input_file = '../data/dialogsum/p1/dialogsum-1p.jsonl'
    # output_path = './output-1p'

    file_name_with_extension = os.path.basename(input_file)
    type, extension = os.path.splitext(file_name_with_extension)

    def segmenting_sentences(idx, data):
        sent_info = []
        for sent in data[idx]['dialogue'].split('\n'):
            sent_info.append(sent)

        os.makedirs(os.path.join(output_path, type), exist_ok=True)

        with open(join(output_path, type, '{}.json'.format(idx)), 'w') as f:
            cur = {}
            cur['dialogue'] = sent_info
            cur['summary'] = data[idx]["summary"]
            json.dump(cur, f, indent=4)


    data = []
    with jsonlines.open(input_file) as reader:
        for line in reader:
            data.append(line)

    n_files = len(data)

    start = time()
    print('{} documents !!!'.format(n_files))

    for i in tqdm.tqdm(range(n_files)):
        segmenting_sentences(i, data)


    # Calculate and print the elapsed time
    elapsed_time = time() - start
    print(f'分句：Time taken: {elapsed_time} seconds')

