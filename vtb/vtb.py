import numpy as np
import enum
from sklearn.metrics import confusion_matrix, accuracy_score
from colorama import init
import seaborn as sn
import matplotlib.pyplot as plt

init()


class POS(enum.IntEnum):
    START = 0
    ADJ = 1
    ADP = 2
    ADV = 3
    AUX = 4
    CCONJ = 5
    DET = 6
    INTJ = 7
    NOUN = 8
    NUM = 9
    PART = 10
    PRON = 11
    PROPN = 12
    PUNCT = 13
    SCONJ = 14
    SYM = 15
    VERB = 16
    X = 17
    END = 18

N = len(POS)

def upostag_convert(tags):
    return [POS.START] + [POS.__dict__[tag] if tag in POS.__dict__.keys() else POS.X for tag in tags] + [POS.END]


def train(data):
    tokens, tags = zip(*data)
    return train_ts(tags), train_os(tokens, tags)


def train_os(tokens, tags):
    counts = np.zeros(N)
    for tag_list in tags:
        for t in tag_list:
            counts[t] += 1

    ob = {}
    for id, tok in enumerate(tokens):
        for i, token in enumerate(tok):
            if token not in ob:
                ob[token] = np.zeros(N)
            ob[token][tags[id][i + 1]] += (1 / counts[tags[id][i + 1]])

    return ob


def train_ts(tags):
    ts = np.zeros((N, N))
    for ulist in tags:
        for i in range(len(ulist) - 1):
            ts[ulist[i], ulist[i + 1]] += 1

    s = np.sum(ts, axis=1)
    s[POS.END] = np.sum(ts[:, POS.END])

    div = np.zeros((N, N))
    for i in range(len(ts)):
        if s[i] > 0:
            div[i][:] = np.divide(ts[i][:], s[i])

    return div


def find_path(B):
    T = B.shape[1] + 1
    path = np.zeros(T, dtype=np.int)
    path[T - 1] = POS.END
    for t in range(T - 1, 0, -1):
        path[t - 1] = B[path[t], t - 1]

    return [POS(t) for t in path]


def apply_vtb(tokenlist, transitions, os):
    unknown_tokens = []
    T = len(tokenlist) + 2
    M = np.zeros((N, T))
    B = np.zeros((N, T - 1), dtype=np.int)

    M[0, 0] = 1

    for t in tokenlist:
        if t not in os:
            os[t] = np.ones(N)
            unknown_tokens.append(t)

    for s in range(1, N):
        M[s, 1] = transitions[0, s] * os[tokenlist[0]][s]

    for t in range(2, T - 1):
        for s in range(1, N - 1):
            state_probs = M[:, t - 1] * transitions[:, s]
            max_prob_state = int(np.argmax(state_probs))
            M[s, t] = state_probs[max_prob_state] * os[tokenlist[t - 1]][s]
            B[s, t - 1] = max_prob_state

    state_probs = M[:, T - 2] * transitions[:, N - 1]
    max_prob_state = int(np.argmax(state_probs))
    M[N - 1, T - 1] = state_probs[max_prob_state]
    B[N - 1, T - 2] = max_prob_state

    return find_path(B), unknown_tokens


def evaluate(data, transitions, os, verbose=False):
    ground_truth, predictions = [], []

    correct_sentences = 0
    unknown_tokens = []
    all_tokens_count = 0

    for idx, item in enumerate(data):
        tokens = item[0]
        tags = item[1]

        all_tokens_count += len(tokens)

        predicted_tags, ut = apply_vtb(tokens, transitions, os)
        unknown_tokens.extend(ut)

        predictions.extend(predicted_tags)
        ground_truth.extend(tags)

        correspond = np.asarray([1 if p == t else 0 for p, t in zip(predicted_tags, tags)])

        correct_sentences += np.prod(correspond)

        if verbose and np.prod(correspond) != 1:
            print(f'tokens: {tokens}')
            print(f'predicted tags      : {predicted_tags[1:-1]}')
            print(f'given     tags      : {tags[1:-1]}')
            print(f'correspondence mask : {correspond[1:-1]}')
            print('sentence accuracy    : {:.2f} %'.format(100 * correct_sentences / (idx + 1)))
            print('{} / {}'.format(correct_sentences, idx + 1))
            print('unknown tokens       : {}'.format(ut))
            print()

    labels = [p.name for p in POS]
    cm = confusion_matrix(ground_truth, predictions, normalize='true')
    for r in range(len(cm)):
        for c in range(len(cm[0])):
            cm[r][c] = round(cm[r][c] , 3)

    plt.figure(figsize=(15, 15))

    sn.heatmap(cm , annot=True ,xticklabels=labels , yticklabels=labels, cbar_kws={"orientation": "horizontal"})

    plt.show()


    sentence_accuracy = correct_sentences / len(data)
    token_accuracy = accuracy_score(ground_truth, predictions)
    unknown_tokens_fraction = len(unknown_tokens) / all_tokens_count

    return [
        'sentence accuracy : {:.2f} %'.format(100 * sentence_accuracy),
        'token    accuracy : {:.2f} %'.format(100 * token_accuracy),
        'unknown  tokens   : {:.2f} %'.format(100 * unknown_tokens_fraction),
        f'{unknown_tokens}'
    ]
