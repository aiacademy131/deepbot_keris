import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

tokenizer = Tokenizer()
# loading
with open('train/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



vocab_size = 10000
model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.load_weights('train/test.h5')


def convert_data(sentence):
  # 토큰화
  sentence = sentence.split(' ')
  # 정수화
  sentence = tokenizer.texts_to_sequences([sentence])
  # 패딩
  sentence = keras.preprocessing.sequence.pad_sequences(sentence,
                                                          value=0,
                                                          padding='post',
                                                          maxlen=50)
  return sentence


def getSentiment(sentence):
  setence = convert_data(sentence)
  result = model.predict(setence)
  return result[0][0]