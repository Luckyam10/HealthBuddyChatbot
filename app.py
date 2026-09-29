"""
app.py
------
Flask backend for the HealthBuddy website.

Endpoints:
    GET  /                         -> serves the frontend
    POST /api/chat                 -> {"message": "..."} -> predictor result
    GET  /api/stats                -> model metrics and active model
    GET  /api/intents              -> supported intents
    GET  /api/confusion-matrix/<kind> -> confusion matrix image

Local:
    python app.py

Production:
    gunicorn app:app
"""

import os
import re

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from model_utils import IntentPredictor


# ---------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

# Allow requests from other origins if the frontend is hosted separately.
CORS(app)

predictor = IntentPredictor()


# ---------------------------------------------------------------------
# Classification report parser
# ---------------------------------------------------------------------

def _parse_classification_report(path):
    """Parse a classification report into structured JSON."""

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    summary = {}

    patterns = [
        ("accuracy", r"Accuracy\s*:\s*([\d.]+)"),
        (
            "precision_macro",
            r"Precision\s*\(macro\)\s*:\s*([\d.]+)"
        ),
        (
            "recall_macro",
            r"Recall\s*\(macro\)\s*:\s*([\d.]+)"
        ),
        (
            "f1_macro",
            r"F1-score\s*\(macro\)\s*:\s*([\d.]+)"
        ),
    ]

    for key, pattern in patterns:
        match = re.search(pattern, content)

        if match:
            summary[key] = float(match.group(1))

    per_class = []

    for line in content.splitlines():

        parts = line.split()

        # Expected format:
        # intent precision recall f1 support
        if len(parts) != 5:
            continue

        label, precision, recall, f1, support = parts

        try:
            per_class.append({
                "intent": label,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            })

        except ValueError:
            continue

    return {
        "summary": summary,
        "per_class": per_class
    }


# ---------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the HealthBuddy frontend."""

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


# ---------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({
            "error": "empty message"
        }), 400

    try:

        result = predictor.predict(message)

        return jsonify(result)

    except Exception as error:

        print(f"Chat prediction error: {error}")

        return jsonify({
            "error": "prediction_failed",
            "message": "The chatbot encountered an error while processing your message."
        }), 500


# ---------------------------------------------------------------------
# Intents API
# ---------------------------------------------------------------------

@app.route("/api/intents")
def intents():

    output = []

    for name, data in predictor.intents.items():

        output.append({
            "intent": name,
            "example_count": len(data["examples"]),
            "sample_examples": data["examples"][:4],
        })

    return jsonify(output)


# ---------------------------------------------------------------------
# Statistics API
# ---------------------------------------------------------------------

@app.route("/api/stats")
def stats():

    semantic_report = _parse_classification_report(
        "classification_report_semantic.txt"
    )

    tfidf_report = _parse_classification_report(
        "classification_report.txt"
    )

    return jsonify({

        "active_model_type": predictor.model_type,

        "semantic_available":
            semantic_report is not None,

        "tfidf_available":
            tfidf_report is not None,

        "semantic_report":
            semantic_report,

        "tfidf_report":
            tfidf_report,

        "confusion_matrix_semantic_url":
            "/api/confusion-matrix/semantic",

        "confusion_matrix_tfidf_url":
            "/api/confusion-matrix/tfidf",

        "intent_count":
            len(predictor.intents),

        "total_examples":
            sum(
                len(data["examples"])
                for data in predictor.intents.values()
            ),
    })


# ---------------------------------------------------------------------
# Confusion matrix API
# ---------------------------------------------------------------------

@app.route("/api/confusion-matrix/<kind>")
def confusion_matrix_image(kind):

    if kind == "semantic":

        filename = "confusion_matrix_semantic.png"

    elif kind == "tfidf":

        filename = "confusion_matrix.png"

    else:

        return jsonify({
            "error": "invalid confusion matrix type"
        }), 400

    if not os.path.exists(filename):

        return jsonify({
            "error": "confusion matrix not found"
        }), 404

    return send_from_directory(
        ".",
        filename
    )


# ---------------------------------------------------------------------
# Local development server
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print(
        f"HealthBuddy website starting. "
        f"Active model: {predictor.model_type}"
    )

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
        threaded=True
    )