import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. Load Dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['cultivar'] = wine.target

# 2. Feature Selection (Selecting exactly 6 features as required)
selected_features = [
    'alcohol', 'malic_acid', 'ash', 
    'magnesium', 'flavanoids', 'color_intensity'
]
X = df[selected_features]
y = df['cultivar']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Pipeline (Includes Mandatory Scaling + Model)
# Using a Pipeline ensures scaling is applied to user input in app.py automatically
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train
model_pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = model_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 7. Save Model (Into the /model/ directory)
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(model_pipeline, 'model/wine_cultivar_model.pkl')
print("Model saved in /model/wine_cultivar_model.pkl")