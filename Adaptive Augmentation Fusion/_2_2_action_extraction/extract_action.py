import os
import json
from time import time
from datetime import timedelta
from os.path import join
from stanfordcorenlp import StanfordCoreNLP
from .extractor import Extractor
from tqdm import tqdm
import nltk

nltk.download('omw-1.4')


# input_file = '../../data/dialogsum/p1/dialogsum-1p.jsonl'
# output_path = 'output-1p'


def extract_events(idx, output_path):
    sent_info = []
    src, event = [], []
    for sent in data[idx]['dialogue'].split('\n'):
        if len(sent) > 1024:
            continue
        info = {}
        src.append(sent)
        info['sentence'] = sent
        info['word'] = nlp.word_tokenize(sent)
        info['pos'] = nlp.pos_tag(sent)
        info['dependency'] = nlp.dependency_parse(sent)
        sent_info.append(info)
    cur_event = extractor.extract(sent_info)
    for j in range(len(cur_event)):
        event.append(' | '.join(cur_event[j]))
    assert len(src) == len(event)
    with open(join(output_path, '{}.json'.format(idx)), 'w') as f:
        cur = {}
        cur['src'] = src
        cur['event'] = event
        cur["summary"] = data[idx]["summary"]
        json.dump(cur, f, indent=4)


def extract_event(input_file, output_path):
    global data, extractor, nlp

    path_to_corenlp = './stanford-corenlp-4.5.7'
    nlp = StanfordCoreNLP(path_to_corenlp)
    extractor = Extractor()

    data = []
    data = []
    with open(input_file, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    n_files = len(data)

    start = time()
    print('extracting events from {} documents !!!'.format(n_files))

    for i in tqdm(range(n_files)):
        extract_events(i, output_path)

    print('finished in {}'.format(timedelta(seconds=time() - start)))
    nlp.close()


def act_ext(input_file, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    extract_event(input_file, output_path)
