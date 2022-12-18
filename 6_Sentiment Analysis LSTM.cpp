import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print("Review : ",x_train[5])
print("Label : ",y_train[5])

vocab=imdb.get_word_index()
print(vocab)

from keras.utils import pad_sequences
max_words=500
x_train = pad_sequences(x_train, maxlen=max_words)
x_test = pad_sequences(x_test, maxlen=max_words)

embedding_size = 32
model = Sequential()
model.add(Embedding(1000, embedding_size, input_length=(max_words)))
model.add(LSTM(100, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# summary of our model.
model.summary()

# Compile the model.
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=128,epochs=10,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

import matplotlib.pyplot as plt
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["accuracy"], color="g", label="Train")
plt.plot(history.history["val_accuracy"], color="b", label="Validation")
plt.legend(loc="best")
plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")
plt.tight_layout()
plt.show()