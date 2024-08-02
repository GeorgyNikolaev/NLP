import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import codecs

import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint, CSVLogger, Callback

input_fname = 'input_file.txt'
output_fname = 'outputfile.txt'
model_fname = 'NLP'
save_name = 'checkpoint.model.keras'
START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'
chars = set([START_CHAR, '\n', PADDING_CHAR, END_CHAR])
with codecs.open(input_fname, "r", "utf-8") as f:
    for line in f:
        chars.update( list(line.strip().lower()) )
char_indices = { c : i for i, c in enumerate(sorted(list(chars))) }
indices_to_chars = { i : c for c, i in char_indices.items() }
num_chars = len(chars)

def get_one(i, size):
    res = np.zeros(size)
    res[i] = 1
    return res

char_vectors = {
    c : (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars))
    for c, v in char_indices.items()
}

sentence_end_markers = set( '.!?' )
sentences = []
current_sentence = ''
with codecs.open(input_fname, "r", "utf-8") as f:
    for line in f:
        s = line.strip().lower()
        if len(s) > 0:
            current_sentence += s + '\n'
        if len(s) == 0 or s[-1] in sentence_end_markers:
            current_sentence = current_sentence.strip()
            if len(current_sentence) > 10:
                sentences.append(current_sentence)
            current_sentence = ''

def get_metrices(sentences):
    max_sentence_len = np.max([len(x) for x in sentences])
    X = np.zeros((len(sentences), max_sentence_len, len(chars)))
    y = np.zeros((len(sentences), max_sentence_len, len(chars)))
    for i, sentence in enumerate(sentences):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(
            max_sentence_len+1, PADDING_CHAR)
        for t in range(max_sentence_len):
            X[i, t, :] = char_vectors[char_seq[t]]
            y[i, t, :] = char_vectors[char_seq[t+1]]
    return X, y


model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(num_chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(clipnorm=1.),
              metrics=['accuracy'])

test_indices = np.random.choice(range(len(sentences)), int(len(sentences) * 0.05))
sentence_train = [ sentences[x]
                   for x in set(range(len(sentences))) - set(test_indices) ]
sentence_test = [ sentences[x] for x in test_indices]
sentence_train = sorted(sentence_train, key = lambda x : len(x))
X_test, y_test = get_metrices(sentence_test)
batch_size = 16
def generate_batch():
    while True:
        for i in range( int(len(sentence_train) / batch_size) ):
            sentences_batch = sentence_train[ i * batch_size : (i+1) * batch_size ]
            yield get_metrices(sentences_batch)

x_train, y_train = get_metrices(sentence_train)

class CharSampler(Callback):
    def __init__(self, char_vectors):
        self.char_vectors = char_vectors

    def on_train_begin(self, logs=None):
        self.epoch = 0
        if os.path.isfile(output_fname):
            os.remove(output_fname)

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample_one(self, T):
        result = START_CHAR
        while len(result) < 100:
            Xsampled = np.zeros( (1, len(result), num_chars) )
            for t, c in enumerate(list(result)):
                Xsampled[0, t, :] = self.char_vectors[c]
            ysampled = self.model(Xsampled)
            yv = ysampled[0, len(result)-1, :]
            selected_char = indices_to_chars[self.sample(yv, T)]
            if selected_char == END_CHAR:
                break
            result = result + selected_char
        return result

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch + 1
        if self.epoch % 50 == 0:
            print("\nEpoch %d text sampling:" % self.epoch)
            with open(output_fname, 'a') as outf:
                outf.write('\n===== Epoch %d =====\n' % self.epoch)
                for T in [0.3, 0.5, 0.7, 0.9, 1.1]:
                    print('\tsampling, T = %.1f...' % T)
                    for _ in range(5):
                        res = self.sample_one(T)
                        outf.write('\nT = %.1f\n%s\n' % (T, res[1:]))


cb_sampler = CharSampler(char_vectors)
cb_logger = CSVLogger(model_fname + '.log')
cb_checkpoint = ModelCheckpoint(filepath=save_name,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True)

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=1000, validation_data=(X_test, y_test),
                    callbacks=[cb_logger, cb_sampler, cb_checkpoint])




