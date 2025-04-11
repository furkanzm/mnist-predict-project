from flask import Flask, request, jsonify, send_from_directory
from train_custom import train_custom_model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import json

app = Flask(__name__)
MODEL_DIR = "saved_models/user"
HISTORY_FILE = "model_history.json"

# Dosyalar eksikse oluştur
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# MNIST test verisi
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = to_categorical(y_test, 10)

# Ana sayfa (demo.html)
@app.route("/")
def index():
    return send_from_directory(".", "demo.html")

@app.route("/tuning_lab.html")
def tuning():
    return send_from_directory(".", "tuning_lab.html")

# TF.js model dosyaları
@app.route("/tfjs_model/<path:filename>")
def serve_tfjs_default(filename):
    return send_from_directory("tfjs_model", filename)

@app.route("/tfjs_model_tuned/<path:filename>")
def serve_tfjs_tuned(filename):
    return send_from_directory("tfjs_model_tuned", filename)

# Custom modeli oluştur
@app.route("/create_model", methods=["POST"])
def create_model():
    params = request.json
    result = train_custom_model(params)

    with open(HISTORY_FILE) as f:
        history_data = json.load(f)
    history_data.append(result)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f, indent=2)

    return jsonify({
        "model_name": result["model_name"],
        "val_accuracy": result["val_accuracy"]
    })

# Kayıtlı modelleri listele
@app.route("/list_models")
def list_models():
    with open(HISTORY_FILE) as f:
        history_data = json.load(f)
    return jsonify([{"model_name": m["model_name"], "created": m["created"]} for m in history_data])

# Modelin başarımını göster
@app.route("/evaluate_model/<model_name>")
def evaluate_model(model_name):
    try:
        model = load_model(os.path.join(MODEL_DIR, f"{model_name}.h5"))
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        return jsonify({"model_name": model_name, "val_accuracy": acc, "loss": loss})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Tüm modelleri kıyasla
@app.route("/compare_models")
def compare_models():
    with open(HISTORY_FILE) as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
