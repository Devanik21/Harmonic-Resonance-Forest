# Save this in examples/basic_usage.py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from HRF_Engine.hrf_eeg import HarmonicResonanceClassifier
from sklearn.metrics import accuracy_score

# 1. Create a dummy dataset
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Basic Workflow
model = HarmonicResonanceClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}")
