from keras import layers, models
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from matplotlib import pyplot

dataset=pd.read_csv("C:/PBL/MODEL/needs.csv",delimiter=",")


X=dataset.iloc[0:,0:5]
Y=dataset.iloc[0:,5]
Yc=pd.get_dummies(Y)

n_train = 7000
trainX, testX = X.iloc[:n_train, :], X.iloc[n_train:, :]
trainY, testY = Yc.iloc[:n_train], Yc.iloc[n_train:]
for i in range(5):
	print('%s => %d ' % (X[i].tolist(), Y[i]))


model = Sequential()
model.add(Dense(50, input_dim=5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='softmax'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history=model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100)

# evaluate the model
_, train_acc = model.evaluate(trainX, trainY)
_, test_acc = model.evaluate(testX, testY)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

predictions = model.predict(X)
predictions_2 = model.predict_classes(X)
for i in range(5):
    print(trainX.iloc[i,:])
    print(trainY.iloc[i,:])
    print(predictions[i])
    print(predictions_2[i])

#  Save model and weights to separated files.
with open("model.json", "w") as file:
     file.write(model.to_json())
model.save_weights("weights.h5")

 # Save model and weights to the same file.
model.save('model.h5', include_optimizer=False)



#
# #X = X.astype('float32')
#
# #X = X.reshape(-1, 28, 28, 1)
# #Y = X.reshape(-1, 28, 28, 1)
#
# #Y= to_categorical(Y)
#
#
# model = models.Sequential()
#model.add(layers.Conv2D(16, 3, activation='relu', input_shape=(1, 6, 1, 1)))
# #model.add(layers.Conv2D(16, 3, activation='relu'))
# model.add(layers.MaxPool2D())
# model.add(layers.Conv2D(32, 3, activation='relu'))
# model.add(layers.MaxPool2D())
# model.add(layers.Conv2D(64, 3, activation='relu'))
# model.add(layers.MaxPool2D())
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(10, activation='softmax'))
# model.summary()
#
# model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
#
# history = model.fit(X, Y, epochs=2, batch_size=128)
#
# print(model.evaluate(X, Y))
#
# # Save model and weights to separated files.
# with open("model.json", "w") as file:
#     file.write(model.to_json())
# model.save_weights("weights.h5")
#
# # Save model and weights to the same file.
# model.save('model.h5', include_optimizer=False)
