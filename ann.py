from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
import numpy as np

(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_test.shape

shuffle_index = np.random.permutation(60000)
x_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train

import matplotlib.pyplot as plt
plt.imshow(X_train[2])

X_train = X_train/255
X_test = X_test/255

X_train[0]

model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=10,validation_split=0.2)

y_prob = model.predict(X_test)

y_pred = y_prob.argmax(axis=1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# plt.imshow(X_test[1])

# model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")