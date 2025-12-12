import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib # For saving the model and pipeline

# --- 1. Load the Dataset ---
print("Loading data...")
try:
    df = pd.read_csv('amazon_delivery.csv')
except FileNotFoundError:
    print("Error: 'amazon_delivery.csv' not found. Please place the dataset in the correct directory.")
    exit()

print("Data loaded successfully.")

# --- 2. Basic Data Cleaning and Preprocessing ---
print("Cleaning and preprocessing data...")

# FIX: Drop rows where the target variable 'Delivery_Time' is missing.
# We cannot train a model on data where the answer is unknown.
df.dropna(subset=['Delivery_Time'], inplace=True)

# Drop the Order_ID as it's just an identifier
df = df.drop('Order_ID', axis=1)

# Handle missing values
# For numerical columns, fill with the median
df['Agent_Age'].fillna(df['Agent_Age'].median(), inplace=True)
df['Agent_Rating'].fillna(df['Agent_Rating'].median(), inplace=True)

# For categorical columns, fill with the mode (most frequent value)
df['Weather'].fillna(df['Weather'].mode()[0], inplace=True)
df['Traffic'].fillna(df['Traffic'].mode()[0], inplace=True)
df['Vehicle'].fillna(df['Vehicle'].mode()[0], inplace=True)
df['Area'].fillna(df['Area'].mode()[0], inplace=True)
df['Category'].fillna(df['Category'].mode()[0], inplace=True)

# Clean text data by removing leading/trailing spaces and standardizing
for col in ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']:
    df[col] = df[col].str.strip()

print("Missing values handled.")


# --- 3. Feature Engineering ---
print("Performing feature engineering...")

# Function to calculate distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers.
    return c * r


# Calculate delivery distance
df['Distance_km'] = haversine(
    df['Store_Latitude'], df['Store_Longitude'],
    df['Drop_Latitude'], df['Drop_Longitude']
)



# Convert time columns to datetime objects
# FIX: Use errors='coerce' to turn any unparseable dates/times into NaT (Not a Time)
df['Order_Time'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'], errors='coerce')
df['Pickup_Time'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Pickup_Time'], errors='coerce')

# Extract time-based features
df['Order_Hour'] = df['Order_Time'].dt.hour
df['Order_Day_of_Week'] = df['Order_Time'].dt.dayofweek # Monday=0, Sunday=6

# FIX: Now, handle any NaNs that might have been created during the datetime conversion
df['Order_Hour'].fillna(df['Order_Hour'].median(), inplace=True)
df['Order_Day_of_Week'].fillna(df['Order_Day_of_Week'].median(), inplace=True)



# For simplicity, we'll drop original date/time/lat-lon columns
df = df.drop([
    'Order_Date', 'Order_Time', 'Pickup_Time',
    'Store_Latitude', 'Store_Longitude',
    'Drop_Latitude', 'Drop_Longitude'
], axis=1)

print("Feature engineering complete.")


# --- 4. Define Features (X) and Target (y) ---
# The target variable is 'Delivery_Time'
X = df.drop('Delivery_Time', axis=1)
y = df['Delivery_Time']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}\n")



# --- 5. Create a Preprocessing Pipeline ---
# This pipeline will handle scaling for numerical data and one-hot encoding for categorical data.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)



# --- 6. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")





# --- 7. Build and Train the Model ---
# We will use a RandomForestRegressor as it's powerful and handles complex relationships well.
print("\nTraining the RandomForestRegressor model...")

# The final model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])





# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training complete.")




# --- 8. Evaluate the Model ---
print("\nEvaluating model performance...")
y_pred = model_pipeline.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")



# --- 9. Save the Model Pipeline ---
# Save the entire pipeline (preprocessor + model) to a file.
pipeline_filename = 'delivery_time_prediction_pipeline.joblib'
joblib.dump(model_pipeline, pipeline_filename)

print(f"\nModel pipeline saved successfully as '{pipeline_filename}'")
