import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    if des is None:
        return np.zeros(128)
    return np.mean(des, axis=0)

def load_dataset(data_dir):
    X, y = [], []
    labels = os.listdir(data_dir)
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    file_path = os.path.join(label_dir, file_name)
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

data_dir = 'root_directory'  # Update this to the path of your dataset
X, y = load_dataset(data_dir)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Save the model and scaler
joblib.dump(model, 'soil_texture_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
