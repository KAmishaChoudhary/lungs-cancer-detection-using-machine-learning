

from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained XGBoost model
model = pickle.load(open('model.pkl', 'rb'))

# Route for the home page
@app.route('/')
def home():
    return render_template('front page.html')

# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data and convert to integers
    int_features = [
        int(request.form['age']),
        int(request.form['gender']),
        int(request.form['pollution']),
        int(request.form['alcohol']),
        int(request.form['dust']),
        int(request.form['hazard']),
        int(request.form['genetics']),
        int(request.form['chronic']),
        int(request.form['diet']),
        int(request.form['obesity']),
        int(request.form['smoking']),
        int(request.form['passive']),
        int(request.form['pain']),
        int(request.form['coughing']),
        int(request.form['fatigue']),
        int(request.form['weight']),
        int(request.form['shortness']),
        int(request.form['wheezing']),
        int(request.form['swallowing']),
        int(request.form['clubbing']),
        int(request.form['frequent']),
        int(request.form['dry']),
        int(request.form['snoring'])
    ]
    
    # Assuming model expects 23 features, pad with zeros if necessary
    n_features_expected = 23
    if len(int_features) < n_features_expected:
        int_features.extend([0] * (n_features_expected - len(int_features)))
    
    # Convert to numpy array and reshape
    final_features = np.array(int_features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Interpret prediction
    if prediction == 0:
        risk = 'LOW'
    elif prediction == 1:
        risk = 'MEDIUM'
    else:
        risk = 'HIGH'
    
    # Render front page.html with prediction result
    return render_template('front page.html', prediction_text=f'Risk Factor: {risk}')

# Entry point for the application
if __name__ == "__main__":
    app.run(debug=True)
