from data_loader import load_data
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    _, _, (x_test, y_test) = load_data()
    model = load_model("saved_model/mnist_ann_extended.h5")

    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test Doğruluğu: {acc * 100:.2f}%")

    y_prob = model.predict(x_test)
    y_pred = y_prob.argmax(axis=1)

    confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karmaşa Matrisi')
    plt.show()

if __name__ == "__main__":
    evaluate_model()