#used env_tf to run the scripts >> requirements.txt

#download the dataset

import pickle
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

with open("fashion_mnist.pkl", "wb") as f:
    pickle.dump((train_images, train_labels, test_images, test_labels), f)

print('Dataset saved as fashion_mnist.pkl')