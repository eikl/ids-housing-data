from flask import Flask, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def test_function():
    data = request.get_json()
    postal_code = data.get('postalCode')
    living_space = data.get('livingSpace')
    apartment_price = data.get('apartmentPrice')
    print(f"postalCode: {postal_code}, livingSpace: {living_space}, apartmentPrice: {apartment_price}")
    return 'Data received!'

if __name__ == '__main__':
    app.run(debug=True)