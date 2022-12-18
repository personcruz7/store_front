from sklearn.neural_network import MLPClassifier
import tensorflow as tf

mnist = tf.keras.datasets.mnist

mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train, y_train)

nsamples, nx, ny = x_train.shape
x2_train = x_train.reshape((nsamples,nx*ny))
# x2_train

nsamples, nx, ny = x_test.shape
x2_test = x_test.reshape((nsamples,nx*ny))
# x2_test

clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(x2_train, y_train)

y_pred=clf.predict(x2_test)
print(clf.score(x2_test, y_test))