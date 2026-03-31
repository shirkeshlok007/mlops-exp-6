import os
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# ---------------------------
# Lazy model loading
# ---------------------------
model = None

def load_model():
    global model
    if model is None:
        import joblib
        model = joblib.load("model.pkl")
    return model

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return "ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    
    data = request.get_json()

    # example input handling
    input_data = np.array(data["input"]).reshape(1, -1)

    prediction = model.predict(input_data)

    return jsonify({
        "prediction": prediction.tolist()
    })

# ---------------------------
# Run locally
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)