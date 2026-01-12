import numpy as np

def predict_heart_disease(model, input_data: list):
    # Convert input to 2D array (sklearn expects 2D)
    arr = np.array(input_data).reshape(1, -1)
    prediction = model.predict(arr)[0]
    return prediction
