import os
from io import open
from conllu import parse_incr
import vtb.vtb as viterbi
import random
import argparse

PATH_ENG_EWT_train = 'data/ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-train.conllu'
PATH_ENG_EWT_test = 'data/ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-test.conllu'

PATH_ENG_GUM_train = 'data/ud-treebanks-v2.5/UD_English-GUM/en_gum-ud-train.conllu'
PATH_ENG_GUM_test = 'data/ud-treebanks-v2.5/UD_English-GUM/en_gum-ud-test.conllu'

def parse_sequence(seq, token_mode='lemma', tag_mode='upostag'):
    tokens = []
    tags = []
    for token in seq.tokens:
        if token[tag_mode] == '_' and token[token_mode] == '_':
            continue

        tokens.append(token[token_mode].lower())
        tags.append(token[tag_mode])

    return tokens, viterbi.upostag_convert(tags)


def parse_dataset(data_path):
    data_file = open(data_path, "r", encoding="utf-8")
    sequence_list_gen = parse_incr(data_file)

    token_lists = []
    tag_lists = []
    for seq in sequence_list_gen:
        tokens, tags = parse_sequence(seq)
        token_lists.append(tokens)
        tag_lists.append(tags)

    return list(zip(token_lists, tag_lists))


def split_data(data, train: float, shuffle=False):
    if shuffle:
        random.shuffle(data)
    N = round(len(data) * train)
    return data[:N], data[N:]


def run():

    if args.ewt:
        train = parse_dataset(PATH_ENG_EWT_train)
        test = parse_dataset(PATH_ENG_EWT_test)
    else:
        train = parse_dataset(PATH_ENG_GUM_train)
        test = parse_dataset(PATH_ENG_GUM_test)

    # train, test = split_data(train + test, 0.2, shuffle=True)
    print('train fraction {:.2f} %'.format(100 * len(train)/(len(train)+len(test))))
    t, o = viterbi.train(train)

    results = viterbi.evaluate(test, t, o, verbose=True)
    print('Evaluation results:')
    print("\n".join(results))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description = "Script for running postagging")
    g1 = arg_parser.add_mutually_exclusive_group(required = True)
    g1.add_argument("--ewt",action='store_true',help="run on EWT dataset")
    g1.add_argument("--gum",action='store_true',help="run on GUM dataset")
    args = arg_parser.parse_args()
    run()

