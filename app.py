"""
app.py — Minimal Flask REST API for the sentiment analysis model.

Endpoints:
    POST /predict        { "texts": ["text1", "text2"] }
    GET  /health
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request
from predict import analyse

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "'texts' list is required"}), 400

    try:
        df = analyse(texts)
        return jsonify(df.to_dict(orient="records"))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[app] Starting Flask API on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
