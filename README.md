# NLP Spam Detection Project

A complete spam detection repository that uses the Enron email dataset, NLP preprocessing, and machine learning classifiers.

## Project Summary

This project explores email spam classification using:

- a large labeled dataset (`enron_spam_data.csv`)
- text cleaning and normalization
- TF-IDF feature extraction
- multiple machine learning classifiers
- a Flask web service for model inference

## Dataset

- File: `enron_spam_data.csv`
- Header: `Message ID,Subject,Message,Spam/Ham,Date`
- Size: ~814,931 CSV lines
- Labels: `spam` and `ham`

## Data Processing Steps

The main notebook (`Project__NLP.ipynb`) performs the following steps:

1. Load and inspect the dataset
2. Create a combined text column for modeling
3. Apply NLP preprocessing using NLTK:
   - lowercase text
   - remove punctuation
   - tokenize words
   - remove English stopwords
   - lemmatize tokens with WordNet
4. Save cleaned text into `df['New_Text']`
5. Convert text into TF-IDF features with:
   - `max_features=5000`
   - `ngram_range=(1, 2)`
   - `min_df=2`
   - `max_df=0.9`
6. Split data into train and test sets with `stratify=y` and `test_size=0.2`

## Models Trained

The notebook trains these classifiers:

- Decision Tree
- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)

## Model Performance (Notebook Results)

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| SVM                | 0.9991   | 0.9983    | 1.0000 | 0.9991   |
| Logistic Regression| 0.9901   | 0.9831    | 0.9977 | 0.9903   |
| Random Forest      | 0.9875   | 0.9844    | 0.9913 | 0.9878   |
| Naive Bayes        | 0.9843   | 0.9779    | 0.9916 | 0.9847   |
| Decision Tree      | 0.9600   | 0.9631    | 0.9581 | 0.9606   |

- Best performing model: **SVM**
- Best F1 score: **0.9991**

## Sample Predictions

The notebook demonstrates the trained models on example text:

- `FREE entry!!! WIN cash now!!! call 09061701461` → `Spam`
- `Don't forget to bring your notebook tomorrow` → `Ham`

## Saved Artifacts

- `spam_model_decision_tree.pkl`
- `spam_model_logistic_regression.pkl`
- `spam_model_naive_bayes.pkl`
- `spam_model_random_forest.pkl`
- `spam_model_svm.pkl`
- `tfidf_vectorizer.pkl`

These files are saved in the repository root and used by the Flask app.

## Flask App (`NLP_spam (1)/`)

This project includes a web service for serving predictions.

### Key files

- `NLP_spam (1)/app.py` - Flask backend
- `NLP_spam (1)/requirements.txt` - app dependencies
- `NLP_spam (1)/index.html` - landing page
- `NLP_spam (1)/docs.html` - app documentation page
- `NLP_spam (1)/models/` - model and vectorizer files used by the app
- `NLP_spam (1)/scan_count.txt` - request counter storage

### App behavior

- Loads `tfidf_vectorizer.pkl` from `NLP_spam (1)/models/`
- Loads saved models from `NLP_spam (1)/models/`
- Returns predictions sorted by model accuracy
- Increments a scan counter for each successful prediction

### Flask API endpoints

- `GET /` → serves `index.html`
- `GET /docs` → serves `docs.html`
- `GET /health` → checks model loading status
- `POST /predict` → returns spam/ham predictions

Example API request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Free money offer"}'
```

## Setup Instructions

### Notebook

1. Open `Project__NLP.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the notebook cells in order.
3. The notebook trains several classifiers and saves the best models to `.pkl` files.

### Flask app

1. Open a terminal in the root project folder.
2. Install dependencies for the app:

```bash
pip install -r "NLP_spam (1)/requirements.txt"
```

3. Run the app:

```bash
python "NLP_spam (1)/app.py"
```

4. Open `http://127.0.0.1:5000/` in a browser.

## Notes

- `NLP_spam (1)/app.py` contains hard-coded reference accuracy values for each model.
- If you want to avoid committing large files, keep generated `.pkl` and dataset files out of version control by using `.gitignore`.
- The dataset is large; preprocessing and training may take time depending on CPU resources.
