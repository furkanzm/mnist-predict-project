from data_loader import load_data, visualize_samples
from model import create_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

def plot_history(history):
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title("Model Training Progress")
    plt.show()

def train_model():
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data()
    visualize_samples(x_train)

    model = create_model()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3)

    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Veri artırma pipeline'ı
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.Reshape((28, 28)),
    ])

    def augment_dataset(x, y):
        x_aug = data_augmentation(x[..., tf.newaxis])
        return x_aug, y

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(32).map(augment_dataset).prefetch(1)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)

    history = model.fit(train_dataset,
                        epochs=10,
                        validation_data=val_dataset,
                        callbacks=[early_stopping, lr_scheduler, tensorboard_cb])

    plot_history(history)

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/mnist_ann_extended.h5")

    print("Model başarıyla eğitildi ve kaydedildi.")
    print(f"TensorBoard logları '{log_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    train_model()