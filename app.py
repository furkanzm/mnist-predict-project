from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

model_paths = {
    "default": "saved_model/mnist_ann.h5",
    "tuned": "saved_model/mnist_tuned.h5"
}

loaded_models = {}

def get_model(name):
    if name not in loaded_models:
        if name in model_paths and os.path.exists(model_paths[name]):
            loaded_models[name] = load_model(model_paths[name])
        else:
            raise ValueError(f"Model '{name}' bulunamadı.")
    return loaded_models[name]

@app.route("/")
def home():
    return "MNIST Prediction API is running. Use /predict?model=default or /predict?model=tuned"

@app.route("/predict", methods=["POST"])
def predict():
    model_key = request.args.get("model", "default")
    try:
        model = get_model(model_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("L").resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            "model": model_key,
            "prediction": predicted_class,
            "probabilities": prediction.tolist()[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)