# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: data_processor.py
# Python  : python3.6
# Time    : 18-9-28 12:15
import nltk
import jieba
import json
import numpy as np
from multiprocessing import Process

with open('../config.json', 'r') as f:
    config = json.load(f)

def process_seq(seq, tokenizer, start=None, end=None, max_num=None, output_and_target=False):
    """# todo: add stop_words?
    process seq and forms voc
    :param seq: input sequence of words: list
    :param tokenizer: default jieba
    :param start: start tokenizer, optional
    :param end: start tokenizer, optional
    :param max_num: limit of the voc size
    :param output_and_target: return two dataset
    :return: dataset and voc
    """
    token_func = {
        "nltk": lambda s: nltk.word_tokenize(s, ),
        "jieba": lambda s: jieba.lcut(s),
        "no": lambda s: [w for w in s]
    }
    X, voc = [], dict()
    for s in seq:
        seq = s.strip().lower().replace(' - ', '-') # todo: add more replacements
        words = token_func[tokenizer](seq)
        if start:
            words.insert(0, start)
        if end:
            words.append(end)
        X.append(words)
        for w in words:
            voc[w] = voc.get(w, 0) + 1

    voc_list = list(sorted([(w, c) for w, c in voc.items()], key=lambda x: x[1], reverse=True))
    max_num = get_max_num(voc_list) if not max_num else max_num
    voc_list = ['<pad>', '<unk>'] + [i[0] for i in voc_list]
    word_voc = {w: i for i, w in enumerate(voc_list[:max_num])}
    len_s = list(sorted([len(s) for s in X]))
    seq_size = len_s[int(len(len_s)*0.9)]
    print('seq_size: {}'.format(seq_size))
    if start and end and output_and_target:
        X1 = [x[:-1] for x in X]
        X2 = [x[1:] for x in X]
        X1 = [[word_voc.get(w, word_voc['<unk>']) for w in x] for x in X1]
        X2 = [[word_voc.get(w, word_voc['<unk>']) for w in x] for x in X2]
        return X1, X2, word_voc, seq_size
    else:
        X_ = [[word_voc.get(w, word_voc['<unk>']) for w in x] for x in X]
        return X_, word_voc, seq_size

def get_max_num(voc_list):
    # 没有设置时的默认字典上限
    count = 0
    total_count = sum(i[1] for i in voc_list)
    for i in range(len(voc_list)):
        count += voc_list[i][1]
        if count / total_count > 0.90: # 出现次数95%以上
            break
    print("voc_size: {}".format(i))
    return i

def gen_datasets():
    X, Y = [], []
    with open('ai_challenger_MTEnglishtoChinese_trainingset_20180827'
              '/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt', 'r') as f:
        for l in f:
            try:
                sentences = l.split('\t')
                X.append(sentences[-2]) # en
                Y.append(sentences[-1]) # cn
            except:
                continue
    encode_input, en_voc, en_seq_size = process_seq(X, 'nltk') # en
    decoder_input, decoder_target, ch_voc, ch_seq_size = process_seq(Y, 'jieba', start='<start>',
                                                                     end='<end>', output_and_target=True)
    config['en_voc_size'] = len(en_voc)
    config['ch_voc_size'] = len(ch_voc)
    config['en_seq_size'] = en_seq_size
    config['ch_seq_size'] = ch_seq_size
    with open('en_voc', 'w') as f:
        json.dump(en_voc, f, ensure_ascii=False, indent=2)
    with open('ch_voc', 'w') as f:
        json.dump(ch_voc, f, ensure_ascii=False, indent=2)
    with open('../config.json', 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    np.save('encode_input', encode_input)
    np.save('decoder_input', decoder_input)
    np.save('decoder_target', decoder_target)

def process_seq_multi(seq, tokenizer, start=None, end=None, max_num=None, output_and_target=False, file=''):
    """# todo: add stop_words?
    process seq and forms voc
    :param seq: input sequence of words: list
    :param tokenizer: default jieba
    :param start: start tokenizer, optional
    :param end: start tokenizer, optional
    :param max_num: limit of the voc size
    :param output_and_target: return two dataset
    :return: dataset and voc
    """
    token_func = {
        "nltk": lambda s: nltk.word_tokenize(s, ),
        "jieba": lambda s: jieba.lcut(s),
        "no": lambda s: [w for w in s]
    }
    X, voc = [], dict()
    for s in seq:
        seq = s.strip().lower().replace(' - ', '-') # todo: add more replacements
        words = token_func[tokenizer](seq)
        if start:
            words.insert(0, start)
        if end:
            words.append(end)
        X.append(words)
    np.save('ai_challenger_MTEnglishtoChinese_trainingset_20180827/{}'.format(file), X)

def gen_dataset_multi(i):
    X, Y = [], []
    with open('ai_challenger_MTEnglishtoChinese_trainingset_20180827/s{}'.format(i), 'r') as f:
        for l in f:
            try:
                sentences = l.split('\t')
                # X.append(sentences[-2])  # en
                Y.append(sentences[-1])  # cn
            except:
                continue
    # process_seq_multi(X, 'nltk', file='x{}'.format(i))  # en
    process_seq_multi(Y, 'jieba', start='<start>', end='<end>', output_and_target=True, file='y{}'.format(i))

def run_multi():
    ps = []
    for i in [1, 4]:
        p = Process(target=gen_dataset_multi, args=(i, ))
        ps.append(p)
    for p in ps:
        p.start()
        print("{} start".format(p.pid))
    for p in ps:
        p.join()
        print("{} joined".format(p.pid))

if __name__ == '__main__':
    run_multi()