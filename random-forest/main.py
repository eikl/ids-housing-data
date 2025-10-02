import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
import joblib
import numpy as np
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def linear_model(x, a, b):
    return a * x + b

df = pd.read_csv('housingdata/src/backend/american_housing_data.csv')
# use only the first 4 numbers of zip codes
# Print the most common zip code
most_common_zip = df['Zip Code'].mode()[0]


print(f'Most common zip code: {most_common_zip}')
print('total unique zip codes:', df['Zip Code'].nunique())
df['Zip Code'] = df['Zip Code'].astype(str).str[:4].astype(int)
print('total unique zip codes:', df['Zip Code'].nunique())
# Remove entries where living area is under 200
df = df[df['Living Space'] >= 200]
# List the 10 most common zip codes and their counts
top_10_zips = df['Zip Code'].value_counts().head(10)
print("Top 10 most common zip codes:")
for zip_code, count in top_10_zips.items():
    print(f"Zip Code {zip_code}: {count} entries")
print()

zip_code = df['Zip Code']
price = df['Price']
living_area = df['Living Space']
beds = df['Beds']
baths = df['Baths']
address = df['Address']
city = df['City']
state = df['State']
zip_code_population = df['Zip Code Population']
zip_code_density = df['Zip Code Density']
county = df['County']
median_household_income = df['Median Household Income']
latitude = df['Latitude']
longitude = df['Longitude']

# Normalize living_area to between 0 and 1
living_area_min = living_area.min()
living_area_max = living_area.max()
living_area = (living_area - living_area_min) / (living_area_max - living_area_min)

price_min = price.min()
price_max = price.max()
price = (price - price_min) / (price_max - price_min)

# Save normalization parameters
normalization_params = {
    'living_area_min': living_area_min,
    'living_area_max': living_area_max,
    'price_min': price_min,
    'price_max': price_max
}
print(f'Living area min: {living_area_min}, max: {living_area_max}')
print(f'Price min: {price_min}, max: {price_max}')


correlation_vars = pd.DataFrame({
    'Price': price,
    'Living Space': living_area,
    'Beds': beds,
    'Baths': baths,
    'Zip Code Population': zip_code_population,
    'Zip Code Density': zip_code_density,
    'Zip Code': zip_code,
    'Median Household Income': median_household_income,
    'Latitude': latitude,
    'Longitude': longitude
})

#random forest parameters:
# predicted variable: price
# predictors: living space, beds, baths, zip code

X = correlation_vars[['Zip Code', 'Living Space', 'Beds', 'Baths']]
y = correlation_vars['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# encoder = TargetEncoder()
# X_train['Zip Code'] = encoder.fit_transform(X_train[['Zip Code']], y_train)
# X_test['Zip Code'] = encoder.transform(X_test[['Zip Code']])

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(X_train.head())

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
RMSE = root_mean_squared_error(y_test, y_pred)

print(f'MAPE: {MAPE}')
print(f'RMSE: {RMSE}')
print(f'R^2: {R2}')
print(f'Mean Squared Error: {mse}')

#predict the price of a house with 2000 sqft, 3 beds, 2 baths, zip code 10036
# Save the trained model to a file
# Save the trained model and normalization parameters to a file
model_data = {
    'model': model,
    'normalization_params': normalization_params
}
joblib.dump(model_data, 'housingdata/src/backend/random_forest_model.pkl')
print("Model and normalization parameters saved to 'random_forest_model.pkl'")
