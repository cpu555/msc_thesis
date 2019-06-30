#  Developed by Morhaf Kourbaj.

from keras.models import model_from_json
from keras.models import load_model
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras_sequential_ascii import sequential_model_to_ascii_printout

class_labels = [
 'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

f = Path("model_structure3.json")
model_structure = f.read_text()

model = model_from_json(model_structure)

model.load_weights("model_weights3.h5")

model=load_model("Adam_elu.h5")
model.summary()
sequential_model_to_ascii_printout(model)
img = image.load_img("jar.jpg.", target_size=(32, 32))

image_to_test = image.img_to_array(img)

list_of_images = np.expand_dims(image_to_test, axis=0)

results = model.predict(list_of_images)
single_result = results[0]
single_result =list(single_result)
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]
class_label = class_labels[most_likely_class_index]
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))

single_result.remove(class_likelihood)
class_labels.remove(class_label)
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]
class_label = class_labels[most_likely_class_index]
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))

single_result.remove(class_likelihood)
class_labels.remove(class_label)
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]
class_label = class_labels[most_likely_class_index]
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))

single_result.remove(class_likelihood)
class_labels.remove(class_label)
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]
class_label = class_labels[most_likely_class_index]
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))

single_result.remove(class_likelihood)
class_labels.remove(class_label)
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]
class_label = class_labels[most_likely_class_index]
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))