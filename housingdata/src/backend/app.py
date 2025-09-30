from flask import Flask, request
from flask_cors import CORS
import joblib
import pandas as pd
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def test_function():
    data = request.get_json()
    postal_code = data.get('postalCode')
    living_space = data.get('livingSpace')
    apartment_price = data.get('apartmentPrice')
    print(f"postalCode: {postal_code}, livingSpace: {living_space}, apartmentPrice: {apartment_price}")

    postal_code = str(postal_code)[:4]  # Use only the first 5 digits

    #
    # PLACEHOLDERS
    #
    beds = 3
    baths = 2

    predicted_price = predict_price(postal_code, living_space, beds, baths)
    #inflation
    predicted_price *= 1.063
    print(f"Predicted Price: {predicted_price}")
    print(f"Postal Code: {postal_code}, Living Space: {living_space}, Beds: {beds}, Baths: {baths}")

    return 'Data received!'

def predict_price(postal_code, living_space, beds, baths):
    # Load the pre-trained model
    model = joblib.load('random_forest_model.pkl')
    # Make a prediction
    feature_names = ['Zip Code', 'Living Space', 'Beds', 'Baths']
    input_data = pd.DataFrame([[postal_code, living_space, beds, baths]], columns=feature_names)
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)