import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the trained classifier
with open('frauddetection.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)

# Route to render the HTML form
@app.route("/")
def home():
    return render_template("prediction_form.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Extract the input values from the form
    features = [int(request.form[field]) for field in ["Year", "Month", "UseChip", "Amount", "MerchantName", "MerchantCity", "MerchantState", "mcc"]]

    try:
        # Make prediction
        prediction = model.predict([features])[0]

        # Convert prediction to human-readable format
        prediction_text = "Fraud" if prediction == 1 else "Not Fraud"
    
        # Return the prediction as a response
        return render_template("prediction_result.html", prediction=prediction_text)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
