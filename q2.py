#run this file in wandb_env >> requirement.txt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Load the dataset
with open("fashion_mnist.pkl", "rb") as f:
    train_images, train_labels, test_images, test_labels = pickle.load(f)

shape=train_images.shape
print(shape)

#Normalizing images
train_images = train_images.reshape(train_images.shape[0],-1)/255.0
test_images = test_images.reshape(test_images.shape[0],-1)/255.0

#one hot encoding
def one_hot_encode(labels,num_classes=10):
    return np.eye(num_classes)[labels]

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

class feedforwardNN:
    

