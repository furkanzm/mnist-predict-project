from data_loader import load_data
import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch
import os

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    # Dinamik gizli katman ve nöron sayısı
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=64, max_value=512, step=64),
            activation=hp.Choice('activation', ['relu', 'tanh']),
            kernel_regularizer=keras.regularizers.l2(hp.Float('l2_reg', 1e-4, 1e-2, sampling='log'))
        ))
        model.add(keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))

    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_tuner():
    (x_train, y_train), (x_valid, y_valid), _ = load_data()

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='mnist_ann_tuning'
    )

    tuner.search(x_train, y_train,
                 epochs=10,
                 validation_data=(x_valid, y_valid),
                 batch_size=32)

    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save("saved_model/mnist_ann_tuned.h5")

if __name__ == "__main__":
    run_tuner()