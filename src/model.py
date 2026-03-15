"""
Machine learning models for fake news detection.
This module implements various ML algorithms and provides training/prediction functionality.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data_processing import prepare_data, DataLoader, FeatureExtractor
import config

class FakeNewsClassifier:
    """Main classifier for fake news detection."""
    
    def __init__(self, model_name: str = "naive_bayes"):
        """
        Initialize classifier with specified model.
        """
        self.model_name = model_name
        self.model = self._create_model(model_name)
        self.feature_extractor = None
        self.is_trained = False
    
    def _create_model(self, model_name: str) -> Any:
        """Create ML model based on name."""
        models = {
            "naive_bayes": MultinomialNB(alpha=1.0),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                max_depth=10,
                min_samples_split=5
            ),
            "logistic_regression": LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000,
                C=1.0
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
        
        return models[model_name]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_extractor: FeatureExtractor) -> None:
        """Train the model."""
        print(f"Training {self.model_name} model...")
        self.model.fit(X_train, y_train)
        self.feature_extractor = feature_extractor
        self.is_trained = True
        print(f"Model training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def predict_text(self, text: str) -> Tuple[int, float]:
        """Predict single text article."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.feature_extractor.transform_texts([text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        return metrics
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Method kept for API compatibility but visualization disabled for production.
        """
        print("Confusion matrix plotting is disabled in production to save memory.")
        pass
    
    def save_model(self, model_path: str, vectorizer_path: str) -> None:
        """Save trained model and vectorizer."""
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_extractor.vectorizer, vectorizer_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
    
    def load_model(self, model_path: str, vectorizer_path: str) -> None:
        """Load trained model and vectorizer."""
        try:
            self.model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.vectorizer = vectorizer
            
            self.is_trained = True
            print(f"Model loaded from: {model_path}")
            print(f"Vectorizer loaded from: {vectorizer_path}")
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
        except Exception as e:
            print(f"Error loading model: {e}")


class ModelComparison:
    """Compare multiple models and select the best one."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_extractor: FeatureExtractor) -> None:
        """Train all available models and evaluate them."""
        print("=== Training and Comparing All Models ===\n")
        
        for model_name in config.MODEL_NAMES:
            print(f"Training {model_name}...")
            classifier = FakeNewsClassifier(model_name)
            classifier.train(X_train, y_train, feature_extractor)
            metrics = classifier.evaluate(X_test, y_test)
            
            self.models[model_name] = classifier
            self.results[model_name] = metrics
            
            print(f"✓ {model_name} - Accuracy: {metrics['accuracy']:.4f}\n")
    
    def get_best_model(self) -> Tuple[str, FakeNewsClassifier]:
        """Get best performing model."""
        if not self.results:
            raise ValueError("No models trained.")
        
        best_model_name = max(self.results.keys(),
                             key=lambda x: self.results[x]['f1_score'])
        return best_model_name, self.models[best_model_name]
    
    def print_comparison(self) -> None:
        """Print comparison of all models."""
        if not self.results:
            print("No models to compare.")
            return
        
        print("=== Model Comparison Results ===")
        for model_name, metrics in self.results.items():
            print(f"{model_name}: {metrics['accuracy']:.4f}")

def train_and_save_best_model(data_file: Optional[str] = None) -> str:
    """Train, select best, and save."""
    loader = DataLoader()
    df = loader.load_data(data_file) if data_file and os.path.exists(data_file) else loader.create_sample_data()
    
    X_train, X_test, y_train, y_test, feature_extractor = prepare_data(df)
    comparison = ModelComparison()
    comparison.train_all_models(X_train, y_train, X_test, y_test, feature_extractor)
    
    best_model_name, best_classifier = comparison.get_best_model()
    model_path = os.path.join(config.MODELS_DIR, "model.pkl")
    vectorizer_path = os.path.join(config.MODELS_DIR, "vectorizer.pkl")
    
    best_classifier.save_model(model_path, vectorizer_path)
    return best_model_name

if __name__ == "__main__":
    try:
        best_model = train_and_save_best_model()
        print(f"Training completed! Best: {best_model}")
    except Exception as e:
        print(f"Error: {e}")