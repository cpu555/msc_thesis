import keras
from keras.datasets import cifar100
from keras.models import model_from_json
from sklearn.externals._arff import xrange
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import numpy as np
# Visualizing of confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils

num_classes = 100 #
f = Path("model_structure3.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights3.h5")

(x_train, y_train), (x_test, y_test) = cifar100.load_data()


y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(x_test, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

for ix in range(100):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)


# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd


df_cm = pd.DataFrame(cm, range(100),
                  range(100))
plt.figure(figsize = (25,25))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.savefig('foo.png')