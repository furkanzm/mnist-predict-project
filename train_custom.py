
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import json, os, datetime

def train_custom_model(params):
    model_name = params["model_name"]
    num_layers = params["num_layers"]
    units = [params["units_0"], params["units_1"], params["units_2"]][:num_layers]
    activation = params["activation"]
    dropout = params["dropout"]
    l2_reg = params["l2_reg"]
    lr = params["lr"]
    epochs = params["epochs"]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for u in units:
        model.add(Dense(u, activation=activation, kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=0)
    val_acc = float(history.history["val_accuracy"][-1])

    save_dir = "custom_saved_models/user"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"{model_name}.h5"))

    return {
        "model_name": model_name,
        "val_accuracy": val_acc,
        "created": datetime.datetime.now().isoformat(),
        "params": params
    }

# Opsiyonel CLI testi
if __name__ == "__main__":
    with open("sample_hyperparams.json") as f:
        sample_params = json.load(f)
    result = train_custom_model(sample_params)
    print(json.dumps(result, indent=2))
