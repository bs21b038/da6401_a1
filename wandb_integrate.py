#run this file in wandb_env >> requirement.txt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import wandb

wandb.init(
    project='fashion_mnist',
    name='plot sample images from each class',
    config={})

# Load the dataset
with open("fashion_mnist.pkl", "rb") as f:
    train_images, train_labels, test_images, test_labels = pickle.load(f)

# Class names for Fashion-MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Get 1 sample per class
unique_classes = np.unique(train_labels)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, label in enumerate(unique_classes):
    index = np.where(train_labels == label)[0][0]  # Find first occurrence
    ax = axes[i // 5, i % 5]
    ax.imshow(train_images[index], cmap="gray")
    ax.set_title(class_names[label])
    ax.axis("off")

plt.tight_layout()
plt.savefig("fashion_mnist_grid.png")  # Save locally

# Log to wandb
wandb.log({"fashion_mnist_grid": wandb.Image("fashion_mnist_grid.png")})

print("Grid image logged to W&B")
