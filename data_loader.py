from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_data():
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    # Normalizasyon
    x_train_full = x_train_full / 255.0
    x_test = x_test / 255.0

    # Doğrulama seti ayrımı
    x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def visualize_samples(x_data):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_data[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()