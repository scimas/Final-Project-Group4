import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
# %% --------------------------------------- Read Data------------------------------------------------------------------
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")
print(train.head())
print(test.head())
print(train.shape)
print(test.shape)

# %% --------------------------------------- Preprocess Data------------------------------------------------------------
labels = train['label'].values
label = LabelBinarizer()
labels = label.fit_transform(labels)
train_images = np.asarray(train.iloc[:,1:])
train_images.shape
train_images = np.expand_dims(train_images,axis = -1)
train_images.shape
train_images = train_images.reshape((-1,28,28))
train_images.shape

# %% --------------------------------------- Train-Test split-----------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(train_images, labels, test_size = 0.3,stratify = labels,random_state = 40)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# %% --------------------------------------- Model ---------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(64, kernel_size=3,activation = 'relu',input_shape=(28, 28 ,1)))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=3,activation = 'relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size =3,activation = 'relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(24, activation = 'softmax'))

# %% --------------------------------------- Summary -------------------------------------------------------------------
LR=0.003
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_check_point = ModelCheckpoint(filepath='CNN.hdf5',
                                    save_best_only=True, save_weights_only=False, monitor = "val_loss")
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=5, batch_size=50,callbacks = [model_check_point])
model.save('CNN.hdf5')
model.summary()

# %% --------------------------------------- Test ----------------------------------------------------------------------
labels = test['label'].values
test_label = LabelBinarizer()
test_labels = test_label.fit_transform(labels)
test_images = np.asarray(test.iloc[:,1:]).astype(np.float32)
test_images = np.expand_dims(test_images,axis = -1)
test_images = test_images.reshape((-1,28,28))
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# %% --------------------------------------- Prediction ----------------------------------------------------------------
y_pred = model.predict(test_images)
accuracy = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_pred, axis=1))
print(accuracy)

cls = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
d = {i:cls[i] for i in range(25) if i!=9}

for i in range(34,44):
    classes = np.argmax(y_pred[i])
    probability = y_pred[i][classes]
    print("Prediction: {} with the probability of: {}".format(d[classes], probability))
    print("class: ", d[labels[i]])
    plt.imshow(test_images[i].reshape(28,28),cmap = 'gray')
    plt.show()
