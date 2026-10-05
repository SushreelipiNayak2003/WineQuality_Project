import joblib
import numpy as np
import os
from flask import Flask, render_template, request


app = Flask(__name__)

# Load trained model
model = joblib.load("model/wine_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = [
        float(request.form["fixed_acidity"]),
        float(request.form["volatile_acidity"]),
        float(request.form["citric_acid"]),
        float(request.form["residual_sugar"]),
        float(request.form["chlorides"]),
        float(request.form["free_sulfur_dioxide"]),
        float(request.form["total_sulfur_dioxide"]),
        float(request.form["density"]),
        float(request.form["pH"]),
        float(request.form["sulphates"]),
        float(request.form["alcohol"])
    ]

    input_data = np.array(values).reshape(1, -1)

    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction=prediction[0]
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)