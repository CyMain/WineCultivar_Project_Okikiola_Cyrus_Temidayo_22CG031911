import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the saved model from the /model/ directory
model = joblib.load('model/wine_cultivar_model.pkl')

# Features we selected during training
FEATURES = ['alcohol', 'malic_acid', 'ash', 'magnesium', 'flavanoids', 'color_intensity']

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect only the 6 required features
        input_values = [float(request.form[f]) for f in FEATURES]
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_values], columns=FEATURES)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Map 0, 1, 2 to Cultivar 1, 2, 3 for better UX
        cultivar_name = f"Cultivar {prediction + 1}"
        
        return render_template('index.html', 
                               features=FEATURES, 
                               prediction_text=f"Predicted Origin: {cultivar_name}")
    except Exception as e:
        return render_template('index.html', features=FEATURES, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)