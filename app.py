from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('credit_card_fraud.pkl')
scaler = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        features = np.array([[
            float(data['V1']),
            float(data['V2']),
            float(data['V3']),
            float(data['V4']),
            float(data['Amount'])
        ]])
        
        # Scale the input features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        if prediction == 1:
            result = 'Fraudulent'
        else:
            result = 'Legit'
            
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
