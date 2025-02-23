from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])

@app.route("/regression", methods=["GET"])
def regression():
    return jsonify(regression)

if __name__ == "__main__":
    app.run(debug=True)
