import pandas as pd
import pickle 
with open("./Model/tree_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Use the loaded model for inference
y_pred_loaded = loaded_model.predict(X_test)

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = predict(X, model)
    if prediction == 1:
        label = "M"
    else:
        label = "B"
    return {
        'status': 200,
        'label': label,
        'prediction': int(prediction)
    }