import numpy as np
import random
import sys

from keras.layers.recurrent import LSTM

from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout

from keras.models import Sequential

class Sequence2Sequence(object):
    def __init__(self, maxlength, chars):
        self.model = self.build_model(maxlength, chars)

    def train(self, X, y, maxlen, chars, char_indices, indices_char, text, nb_epochs):
        for iteration in range(1, nb_epochs):
            print()
            print('-' * 50)
            print('Iteration', iteration)

            self.model.fit(X, y, batch_size=128, nb_epoch=1)

        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = self.model.predict(x, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()



    def build_model(self, maxlength, chars_len):
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, input_shape=(maxlength, chars_len)))
        model.add(Dropout(0.2))
        model.add(LSTM(512, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(chars_len))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        return model

    def sample(self, a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

