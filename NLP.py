# Код, чтобы в начале не выдавало два предупреждения.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import codecs
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint, CSVLogger, Callback


# Название файлов.
input_fname = 'input_file.txt'
output_fname = 'outputfile.txt'
model_fname = 'NLP'
save_name = 'checkpoint.model.keras'

# Гиперпараметры.
batch_size = 16
clipnorm = 1.
learning_rate = 0.001

# Начальный и конечный символы. А также символ для заполнения предложений.
START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'
chars = set([START_CHAR, '\n', PADDING_CHAR, END_CHAR])  # Множество со всеми возможными символами.

# Построчное считывание файла и заполнения списка chars.
with codecs.open(input_fname, "r", "utf-8") as f:
    for line in f:
        chars.update(list(line.strip().lower()))

# Словари символ - индекс и индекс - символ.
char_indices = {c: i for i, c in enumerate(sorted(list(chars)))}
indices_to_chars = {i: c for c, i in char_indices.items()}
num_chars = len(chars)  # количество используемых символов.


# Функция для возвращения one-hot вектора.
def get_one(i, size):
    res = np.zeros(size)
    res[i] = 1
    return res


# Словарь с one-hot векторами.
char_vectors = {
    c: (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars))
    for c, v in char_indices.items()
}

sentence_end_markers = set('.!?')  # Множество маркеров конца предложения.
sentences = []  # Массив со всем предложениями.
current_sentence = ''  # Текущее предложение.

# Считывание и добавление строчек длинной более 10 в sentences.
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


# Функция для создания входа и выхода.
# Вход и выход отличаются сдвигом на 1 символ
def get_metrices(sentences):
    # Нахождение длины самого длинного предложения,
    # Чтобы все предложения были одинаковой длины.
    max_sentence_len = np.max([len(x) for x in sentences])

    # Массив входа(X) и выхода (y) с предложениями,
    # Где каждый символ предложения это one-hot вектор.
    X = np.zeros((len(sentences), max_sentence_len, len(chars)))
    y = np.zeros((len(sentences), max_sentence_len, len(chars)))
    for i, sentence in enumerate(sentences):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(
            max_sentence_len+1, PADDING_CHAR)
        for t in range(max_sentence_len):
            X[i, t, :] = char_vectors[char_seq[t]]
            y[i, t, :] = char_vectors[char_seq[t+1]]
    return X, y


# Описание архитектуры модели NLP.
model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(num_chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(clipnorm=clipnorm, learning_rate=learning_rate),
              metrics=['accuracy'])

# Список индексов для тестового множества.
test_indices = np.random.choice(range(len(sentences)), int(len(sentences) * 0.05))

# Тренировочное и тестовое множества.
sentence_train = [sentences[x] for x in set(range(len(sentences))) - set(test_indices)]
sentence_test = [sentences[x] for x in test_indices]
sentence_train = sorted(sentence_train, key=lambda x: len(x))
x_train, y_train = get_metrices(sentence_train)
x_test, y_test = get_metrices(sentence_test)


# Класс для Callback.
class CharSampler(Callback):
    def __init__(self, char_vectors):
        self.char_vectors = char_vectors

    def on_train_begin(self, logs=None):
        self.epoch = 0
        if os.path.isfile(output_fname):
            os.remove(output_fname)

    # Выбор символа методом сэмплирование.
    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # Порождение одного сэмпла.
    def sample_one(self, T):
        result = START_CHAR

        # Посимвольное порождение текста.
        while len(result) < 100:
            Xsampled = np.zeros((1, len(result), num_chars))
            for t, c in enumerate(list(result)):
                Xsampled[0, t, :] = self.char_vectors[c]
            ysampled = self.model(Xsampled)
            yv = ysampled[0, len(result)-1, :]
            selected_char = indices_to_chars[self.sample(yv, T)]
            if selected_char == END_CHAR:
                break
            result = result + selected_char
        return result

    # Каждые 50 эпох в файл записывается по 5 примеров
    # Для каждого значения температуры.
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


# Callbacks.
cb_sampler = CharSampler(char_vectors)
cb_logger = CSVLogger(model_fname + '.log')
cb_checkpoint = ModelCheckpoint(filepath=save_name,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True)

# Обучение.
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=1000, validation_data=(x_test, y_test),
                    callbacks=[cb_logger, cb_sampler, cb_checkpoint])




