from flask import Flask, jsonify
from src.backend.models.linear_regression import results
from src.backend.models.random_forest import rf_results

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route("/regression", methods=["GET"])
def regression_results():
    return jsonify(results)

@app.route("/random_forest", methods=["GET"])
def random_forest_results():
    return jsonify(rf_results)

if __name__ == "__main__":
    app.run(debug=True)
