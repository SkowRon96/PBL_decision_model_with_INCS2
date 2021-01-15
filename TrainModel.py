from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
from keras import layers, models
from keras.models import load_model
import tensorflow as tf
import pandas as pd

dataset=pd.read_csv("needs.csv",delimiter=",")

X=dataset.iloc[0:,0:5]
Y=dataset.iloc[0:,5]
Yc=pd.get_dummies(Y)

#for i in range(5):
#	print('%s => %d ' % (X[i].tolist(), Y[i]))

n_train = 7000
trainX, testX = X.iloc[:n_train, :], X.iloc[n_train:, :]
trainY, testY = Yc.iloc[:n_train], Yc.iloc[n_train:]

model = Sequential()
model.add(Dense(50, input_dim=5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='softmax'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history=model.fit(trainX, trainY, validation_data=(testX, testY), epochs=25)

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
with open("Keras_Model/model.json", "w") as file:
     file.write(model.to_json())
model.save_weights("Keras_Model/weights.h5")

 # Save model and weights to the same file.
model.save('Keras_Model/model.h5', include_optimizer=False)


