import json
import os
import Model as models

import numpy as np
import sys
import getopt
import random
import pickle

from sklearn.feature_extraction.text import CountVectorizer

def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['nb_epochs=', 'sample_size='])
    except getopt.GetoptError as error:
        print 'error'
        sys.exit(2)

    nb_epochs = 60
    sample_size = 8000
    for opt, arg in opts:
        if opt == '--nb_epochs':
            nb_epochs = int(arg)
        elif opt == '--sample_size':
            sample_size = int(arg)

    channels = read_messages('LAN Slack')
    text = []

    for channel in channels:

        for file in channel:
            with open(file) as messages:

                messages = json.load(messages)


                for message in messages:
                    rand = np.random.randint(low=0, high=10)
                    if rand >= 8:
                        if 'text' in message.keys():
                            text.append(message['text'] + "\n")

    sample = np.random.choice(text, sample_size, replace=False)
    text = "".join(sample)

    print "Length of text: {}".format(len(text))
    maxlen = 40
    step = 3

    X, y, sentences, indices_char, char_indices, chars = preprocess_text(text, maxlen, step)

    s2s = models.Sequence2Sequence(maxlen, len(chars))
    s2s.train(X, y, maxlen, chars, char_indices, indices_char, text, nb_epochs)
    pickle.dump(chars, open('chars.p', 'w+'))
    pickle.dump(char_indices, open('char_i.p', 'w+'))
    pickle.dump(indices_char, open('i_char.p', 'w+'))
    pickle.dump(text, open('text.p', 'w+'))


def read_messages(dir):
    subdirs = os.listdir(dir)
    channels = []

    for subdir in subdirs:
        files = get_all_files(dir + '/' + subdir)
        channels.append(files)

    return channels

def get_all_files(path):
    file_paths = []

    for path, subdirs, files in os.walk(path):
        for name in files:

            # Make sure hidden files do not make into the list
            if name[0] == '.':
                continue
            file_paths.append(os.path.join(path, name))


    return file_paths

def preprocess_text(text, maxlen, step):
    chars = set(text)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters

    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')

    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return X, y, sentences, indices_char, char_indices, chars

if __name__ == '__main__':
    main()

