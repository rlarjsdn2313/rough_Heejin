import numpy

import tensorflow as tf

from numpy import array

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.python.keras.engine.training import flatten_metrics_in_order


docs = []
with open('data', 'r') as f:
    docs = f.read().split('\n')

answer = []
for d in docs:
    answer.append(d[0])

for i in range(len(docs)):
    docs[i] = docs[i][2:]

classes = array(answer)

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)

x = token.texts_to_sequences(docs)

padded_x = pad_sequences(x, 15)
print('\n패딩 결과\n', padded_x)

word_size = len(token.word_index) + 1

model = Sequential()
model.add(Embedding(word_size, 8, input_length=15))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x, classes, epochs=20)
print('\n Accuracy : %4f' % (model.evaluate(padded_x, classes)[1]))
