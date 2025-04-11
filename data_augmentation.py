from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_cifar10():
    (x_train, y_train), (_, _) = cifar10.load_data()
    return x_train / 255.0, y_train

def visualize_augmented(image):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented = data_augmentation(tf.expand_dims(image, 0))
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented[0])
        plt.axis("off")
    plt.show()

def run_data_augmentation():
    x_train, _ = load_cifar10()
    image = x_train[12]
    visualize_augmented(image)

if __name__ == "__main__":
    run_data_augmentation()