import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load data
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# Reshape X to 3D for 1D CNN: (samples, steps, features) -> (samples, 12, 1)
X = np.expand_dims(X, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build a lightweight 1D CNN feature extractor
cnn_model = models.Sequential([
    layers.Input(shape=(12, 1)),
    layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(), # Flattens features into a clean embedding vector
])

# 3. Extract deep embeddings from the CNN
print("Extracting Deep Features using 1D CNN...")
X_train_features = cnn_model.predict(X_train)
X_test_features = cnn_model.predict(X_test)

# 4. Train the SVM Classifier on top of the deep features
print("Training Hybrid Model: 1D CNN features + SVM Backend...")
svm_classifier = SVC(kernel='rbf', C=1.0)
svm_classifier.fit(X_train_features, y_train)

# 5. Evaluate Hybrid Pipeline
y_pred = svm_classifier.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nHybrid Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))