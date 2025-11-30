from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # <-- Critical

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS":
        # Preflight response
        response = jsonify({"message": "CORS preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    # Actual POST prediction
    data = request.get_json()
    text = data.get("text", "")
    
    print("Received text:", text)

    df = pd.DataFrame([{
        "category": "",
        "rating": 0,
        "text_": text
    }])
    
    prediction = model.predict(df)[0]

    if prediction == "OR":
        prediction = "Original Review"
    else:
        prediction = "AI Generated Review"
        
    response = jsonify({"prediction": prediction})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    app.run(port=5000, debug=True)
