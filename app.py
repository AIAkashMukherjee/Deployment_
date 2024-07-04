import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model_path = 'models/regmodel.pkl'
scale_path = 'models/scaling.pkl'

try:
    loaded_model = pickle.load(open(model_path, 'rb'))
    scaled_model = pickle.load(open(scale_path, 'rb'))
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    loaded_model = None
    scaled_model = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None
    scaled_model = None

def predictions(loaded_model, scaled_model, MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    if loaded_model is None or scaled_model is None:
        return "Model not loaded"

    input_data = {
        'MedInc': [MedInc],
        'HouseAge': [HouseAge],
        'AveRooms': [AveRooms],
        'AveBedrms': [AveBedrms],
        'Population': [Population],
        'AveOccup': [AveOccup],
        'Latitude': [Latitude],
        'Longitude': [Longitude]
    }
    input_df = pd.DataFrame(input_data)

    # Scale the input data
    scaled_input = scaled_model.transform(input_df)

    # Predict using the loaded model
    predicted_price = loaded_model.predict(scaled_input)

    return predicted_price[0]

def main():
    st.title('House Price Prediction System')
    st.header("Please enter the details to predict the house price")

    MedInc = st.number_input("MedInc", min_value=0.49, max_value=15.00)
    HouseAge = st.number_input("HouseAge", min_value=1.0, max_value=52.0)
    AveRooms = st.number_input("Average Rooms", min_value=0.84, max_value=141.0)
    AveBedrms = st.number_input("Average Bedrooms", min_value=1.0, max_value=10.0)
    Population = st.number_input("Population", min_value=3.00, max_value=35682.00)
    AveOccup = st.number_input("AveOccup", min_value=0.69, max_value=1243.33)
    Latitude = st.number_input("Latitude", min_value=32.54, max_value=41.95)
    Longitude = st.number_input("Longitude", min_value=-124.35, max_value=-114.31)

    if st.button('Predict'):
        result = predictions(loaded_model, scaled_model, MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
        st.success(f"The predicted house price is: ${result:.2f}")

if __name__ == '__main__':
    main()
