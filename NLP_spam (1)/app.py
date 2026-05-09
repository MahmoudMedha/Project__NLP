import os
import threading

from flask import Flask, jsonify, request, send_from_directory

import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SCAN_COUNT_PATH = os.path.join(BASE_DIR, "scan_count.txt")

vectorizer = None
vectorizer_error = None
models = {}
model_files = []
model_errors = {}

MODEL_NAME_BY_FILE = {
    "spam_model_svm.pkl": "SVM",
    "spam_model_logistic_regression.pkl": "Logistic Regression",
    "spam_model_random_forest.pkl": "Random Forest",
    "spam_model_naive_bayes.pkl": "Naive Bayes",
    "spam_model_decision_tree.pkl": "Decision Tree",
}

ACCURACY_BY_MODEL = {
    "SVM": 0.992438,
    "Logistic Regression": 0.990065,
    "Random Forest": 0.987248,
    "Naive Bayes": 0.984282,
    "Decision Tree": 0.959964,
}

_scan_lock = threading.Lock()
total_scans = 0


def normalize_label(y):
    y = int(y)
    if y == 1:
        return "spammmmm"
    return "ham"


def display_name_from_filename(filename):
    if filename in MODEL_NAME_BY_FILE:
        return MODEL_NAME_BY_FILE[filename]
    stem = os.path.splitext(filename)[0]
    if stem.startswith("spam_model_"):
        stem = stem[len("spam_model_"):]
    return stem.replace("_", " ").title()


def discover_model_files(models_dir):
    results = []
    try:
        filenames = os.listdir(models_dir)
    except Exception:
        return []

    for filename in filenames:
        if not (filename.startswith("spam_model_") and filename.endswith(".pkl")):
            continue
        name = display_name_from_filename(filename)
        results.append((name, os.path.join(models_dir, filename)))

    return sorted(results, key=lambda x: x[0].lower())


def read_scan_count(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return int(raw) if raw else 0
    except FileNotFoundError:
        return 0
    except Exception:
        return 0


def write_scan_count(path, value):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(str(int(value)))
    os.replace(tmp_path, path)


total_scans = read_scan_count(SCAN_COUNT_PATH)


def load_artifacts():
    global vectorizer, vectorizer_error, models, model_files, model_errors
    vectorizer = None
    vectorizer_error = None
    models = {}
    model_errors = {}

    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        vectorizer_error = str(e)

    model_files = discover_model_files(MODELS_DIR)
    if vectorizer is None:
        return

    for name, path in model_files:
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            model_errors[name] = str(e)


load_artifacts()


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/docs")
def docs():
    return send_from_directory(BASE_DIR, "docs.html")


@app.get("/health")
def health():
    ok = vectorizer is not None and len(models) > 0
    payload = {
        "ok": ok,
        "total_scans": total_scans,
        "models_loaded": sorted(list(models.keys())),
        "models_failed": model_errors,
    }
    if vectorizer_error:
        payload["error"] = vectorizer_error
    return jsonify(payload), (200 if ok else 500)


@app.post("/predict")
def predict():
    if vectorizer is None:
        return jsonify({"ok": False, "error": vectorizer_error or "Vectorizer not loaded"}), 500

    if len(model_files) == 0:
        return jsonify({"ok": False, "error": "No models found in models directory"}), 500

    body = request.get_json(silent=True) or {}
    text = body.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"ok": False, "error": "Send JSON like: {'text': '...'}"}), 400

    X = vectorizer.transform([text])
    predictions = []
    for name, _path in model_files:
        accuracy = ACCURACY_BY_MODEL.get(name)
        if name not in models:
            predictions.append({
                "model": name,
                "accuracy": accuracy,
                "label": None,
                "error": model_errors.get(name, "Model not loaded"),
            })
            continue

        try:
            y = models[name].predict(X)[0]
            predictions.append({
                "model": name,
                "accuracy": accuracy,
                "label": normalize_label(y),
            })
        except Exception as e:
            predictions.append({
                "model": name,
                "accuracy": accuracy,
                "label": None,
                "error": str(e),
            })

    predictions = sorted(
        predictions,
        key=lambda r: (r.get("accuracy") is not None, r.get("accuracy") or 0.0),
        reverse=True,
    )
    best_row = next((r for r in predictions if r.get("label")), None)

    if best_row is None:
        return jsonify({"ok": False, "error": "All models failed to predict", "predictions": predictions}), 500

    global total_scans
    with _scan_lock:
        total_scans += 1
        try:
            write_scan_count(SCAN_COUNT_PATH, total_scans)
        except Exception:
            pass

    return jsonify({"ok": True, "label": best_row["label"], "predictions": predictions, "total_scans": total_scans}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
