from flask import Flask, request, jsonify
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import os 

app = Flask(__name__)

loaded_model = XGBClassifier()
loaded_model.load_model('Model/xgb_model.json')

failure = { 0 :"No failure.The Milling Machine is perfroming well. " , 
           1 :"Tool Wear Failure.Failure since tool wear time is between 200 - 240 mins.It has reached a critical level." ,
           2 : "Heat dissipation failure.Indicates a failure due to inadequate heat dissipation, typically caused by low air-process temperature difference and/or low tool rotational speed." ,
           3:"Power Failure.Indicates a failure caused by insufficient or excessive power for the milling process.The power is below 3500 W or above 9000 W.",
           4 : "Overstrain Failure.Indicates a failure due to excessive strain or stress on the milling machine, potentially leading to mechanical damage." 
           , 5 : "Random Failures.Failure that occur sporadically and cannot be attributed to a specific cause." }
# Define a route for making predictions
@app.route('/predict_failure', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Make predictions
    prediction = loaded_model.predict(processed_data)
    
    # Map prediction to failure type
    failure_type = failure[int(prediction[0])]
    
    # Return the prediction as JSON response
    return jsonify({'prediction': failure_type })
def preprocess_data(data):
    # Convert JSON data to DataFrame
    X = pd.DataFrame(data, index=[0])
    
    # Change columns to float
    for column in X.columns:
        try:
            X[column] = X[column].astype(float)
        except:
            pass
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X[X.select_dtypes(include=['float64']).columns] = imputer.fit_transform(X.select_dtypes(include=['float64']))
    
    # Perform feature engineering
    X['Power'] = X['Rotational speed [rpm]'] * X['Torque [Nm]']
    X['Temperature difference'] = X['Process temperature [K]'] - X['Air temperature [K]']
    
    X['Type_L'] = (X['Type'] == 'L').astype(int)
    X['Type_M'] = (X['Type'] == 'M').astype(int)
    X.loc[X['Type'] == 'S', ['Type_M', 'Type_L']] = 0
    X.drop(['UDI', 'Product ID','Type'], axis=1, inplace=True)
    
    # Rename columns to remove problematic characters
    X.rename(columns=lambda x: x.replace('[', '_').replace(']', '_').replace('<', '_'), inplace=True)
    
    return X

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT"))
