import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load pre-extracted data
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# 2. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Initialize and train Random Forest
print("Training Traditional Model: Chroma + Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTraditional Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Export Model for the Web App
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("-> Successfully exported 'random_forest_model.pkl'")