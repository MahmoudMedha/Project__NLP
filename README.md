# NLP Spam Detection Project

A spam classification project using natural language processing and machine learning models trained on the Enron email dataset.

## Repository Structure

- `Project__NLP.ipynb` - Main Jupyter notebook with data exploration, preprocessing, feature engineering, model training, and evaluation.
- `enron_spam_data.csv` - The dataset containing labeled spam/ham email examples.
- `NLP_spam (1)/` - Flask web app, documentation, and supporting files.
  - `app.py` - Flask API for serving predictions from saved models.
  - `index.html` - App landing page.
  - `docs.html` - App documentation page.
  - `requirements.txt` - Python dependencies for the web app.
  - `models/` - Saved model artifacts and vectorizer used by the app.
  - `scan_count.txt` - Persistent request counter for the web app.
- `spam_model_decision_tree.pkl` - Saved decision tree model.
- `spam_model_logistic_regression.pkl` - Saved logistic regression model.
- `spam_model_naive_bayes.pkl` - Saved Naive Bayes model.
- `spam_model_random_forest.pkl` - Saved random forest model.
- `spam_model_svm.pkl` - Saved SVM model.
- `tfidf_vectorizer.pkl` - Saved TF-IDF text vectorizer.

## What's Included

- Multiple classification algorithms for spam detection
- A reusable `TfidfVectorizer` for text preprocessing
- A Flask API that loads models from `NLP_spam (1)/models/`
- A small web UI and documentation pages

## Running the Notebook

1. Open `Project__NLP.ipynb` in Jupyter Notebook or JupyterLab.
2. Execute the cells to inspect the data, preprocess the text, train classifiers, and evaluate their performance.

## Running the Flask App

1. Open a terminal in `NLP_spam (1)/`.
2. Install dependencies:

```bash
pip install -r "NLP_spam (1)/requirements.txt"
```

3. Start the app:

```bash
python "NLP_spam (1)/app.py"
```

4. Open `http://127.0.0.1:5000/` in a browser.

## API Endpoints

- `GET /` - Serves `index.html`
- `GET /docs` - Serves `docs.html`
- `GET /health` - Returns app health and loaded model status
- `POST /predict` - Predicts spam/ham from JSON payload

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Free money offer"}'
```

## Notes

- The app expects `tfidf_vectorizer.pkl` and `spam_model_*.pkl` files in `NLP_spam (1)/models/`.
- If you want to keep large datasets or model binaries out of version control, add them to `.gitignore`.
