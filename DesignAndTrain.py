#  Developed by Morhaf Kourbaj.

import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras_sequential_ascii import sequential_model_to_ascii_printout

RUN_NAME = "2nd Adam Run relu mmmm"
logger = keras.callbacks.TensorBoard(
    log_dir='Final/{}'.format(RUN_NAME),
    write_graph=True,
    histogram_freq=5
)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)


def base_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu", name="Conv1"))
    model.add(Conv2D(32, (3, 3), activation="relu", name="Conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool1"))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu", name="Conv3"))
    model.add(Conv2D(64, (3, 3), activation="relu", name="Conv4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool2"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu", name="Dense1"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="softmax", name="Dense2"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

cnn_n = base_model()
cnn_n.summary()

# Viewing model structure

sequential_model_to_ascii_printout(cnn_n)


my_network = cnn_n.fit(
    x_train,
    y_train,
    batch_size=2048,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[logger],
    shuffle=True
)

cnn_n.save('Adam_relummm.h5')

scores = cnn_n.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



# Plots for training and testing process: loss and accuracy

plt.figure(0)
plt.plot(my_network.history['acc'],'r')
plt.plot(my_network.history['val_acc'],'g')
plt.xticks(np.arange(0, 52, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])


plt.figure(1)
plt.plot(my_network.history['loss'],'r')
plt.plot(my_network.history['val_loss'],'g')
plt.xticks(np.arange(0, 52, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
plt.show()

# Confusion matrix result

from sklearn.metrics import classification_report, confusion_matrix
Y_pred = cnn_n.predict(x_test, verbose=2)
y_pred = np.argmax (Y_pred, axis = 1 )

for ix in range(100):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd


df_cm = pd.DataFrame(cm, range(100),
                  range(100))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.savefig('Adam_relummm.png')