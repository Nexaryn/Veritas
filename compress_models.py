import joblib
import os

# List all your model files
files = [
    'models/logistic_regression_model.joblib',
    'models/naive_bayes_model.joblib',
    'models/random_forest_model.joblib',
    'models/svm_model.joblib',
    'models/vectorizer.joblib'
]

for f in files:
    if os.path.exists(f):
        m = joblib.load(f)
        joblib.dump(m, f, compress=3)
        print(f"Shrunk {f}")