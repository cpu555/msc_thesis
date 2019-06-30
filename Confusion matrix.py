#  Developed by Morhaf Kourbaj.

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


f = Path("model_structure3.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights3.h5")

(x_train, y_train), (x_test, y_test) = cifar100.load_data()


def return_label(one_hot_lab):
    op = np.zeros(len(one_hot_lab))
    for i in xrange(len(one_hot_lab)):
        op[i] = np.argmax(one_hot_lab[i])
    return op

# Generate Confusion Matrix:
def prep_confusion_matrix(y_test, y_pred, num_of_classes=100):
    op = np.zeros((num_of_classes, num_of_classes))
    for i in xrange(len(y_test)):
        op[int(y_test[i]),int(y_pred[i])] += 1
    return op

# Getting the predictions
predictions = model.predict(x_test, verbose=1)

# Getting the predicted labels
pred_labels = return_label(predictions)
print(pred_labels)
# De-one-hotting labels
y_tst_dh = return_label(y_test)

# Preparing the confusion Matrix
conf_mat = prep_confusion_matrix(y_tst_dh, pred_labels)

# Plotting the Confusion Matrix
df_cm = pd.DataFrame(conf_mat, index = [i for i in xrange(100)],
                  columns=[i for i in xrange(100)])
plt.figure(figsize=(25, 25))
sn.heatmap(df_cm, annot=True)
plt.savefig('foo.png')