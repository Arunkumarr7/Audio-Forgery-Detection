import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models
import pickle

# 1. Load data
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# Reshape X to 3D for 1D CNN: (samples, 12, 1)
X = np.expand_dims(X, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Build a Trainable 1D CNN + Classification Head
cnn_model = models.Sequential([
    layers.Input(shape=(12, 1)),
    layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(), 
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

# 3. Pre-train the CNN layers to optimize feature extraction boundaries
print("Pre-training 1D CNN layers...")
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# 4. FIXED: Create the Feature Extractor by slicing the trained layers directly
print("\nExtracting Optimized Deep Features using Trained 1D CNN...")
feature_extractor = models.Sequential(cnn_model.layers[:5])

X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# 5. Train the SVM Classifier Backend
print("Training Hybrid Model: Trained 1D CNN features + SVM Backend...")
svm_classifier = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
svm_classifier.fit(X_train_features, y_train)

# 6. Evaluate
y_pred = svm_classifier.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nHybrid Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Export Model Components for the Web App
feature_extractor.save('cnn_feature_extractor.h5')
with open('svm_classifier.pkl', 'wb') as f:
    pickle.dump(svm_classifier, f)
print("-> Successfully exported 'cnn_feature_extractor.h5' and 'svm_classifier.pkl'")