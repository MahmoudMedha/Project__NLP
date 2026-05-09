# NLP Spam Detection Project

This repository contains an NLP-based spam detection project built on the Enron spam dataset.

## Contents

- `Project__NLP.ipynb` - Main notebook with exploration, preprocessing, and model training.
- `enron_spam_data.csv` - Dataset used for spam classification.
- `NLP_spam (1)/` - Auxiliary folder with app files, documentation, and additional notebooks.
- `spam_model_*.pkl` - Exported trained models.
- `tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer for text transformation.

## Usage

1. Open `Project__NLP.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells to explore the dataset, preprocess text, train models, and evaluate performance.
3. Use the saved `.pkl` models and vectorizer for inference or deployment.

## Requirements

Install the required Python packages listed in `NLP_spam (1)/requirements.txt` if available, or use the following as a starting point:

```bash
pip install numpy pandas scikit-learn jupyter
```

## Notes

- Keep dataset files and model artifacts under version control only if needed; otherwise use `.gitignore` to avoid committing large or generated files.
- The `NLP_spam (1)/` subfolder contains a small app and static documentation.
