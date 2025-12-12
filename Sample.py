import joblib
import pandas as pd
print("Loading the saved model pipeline...")
try:
    # The filename must match the one you saved earlier
    pipeline_filename = 'delivery_time_prediction_pipeline.joblib'
    loaded_pipeline = joblib.load(pipeline_filename)
except FileNotFoundError:
    print(f"Error: Model file '{pipeline_filename}' not found.")
    print("Please make sure you have run 'delivery_model_creation.py' first to create the model file.")
    exit()

print("Model loaded successfully!")

new_order_data = {
    'Agent_Age': [55],
    'Agent_Rating': [3],
    'Weather': ['Sunny'],
    'Traffic': ['Low'],
    'Vehicle': ['cycle'],
    'Area': ['Urban'],
    'Category': ['Electronic'],
    # Raw features that will be engineered
    'Store_Latitude': [28.8386],
    'Store_Longitude': [78.7733],
    'Drop_Latitude': [28.8480],
    'Drop_Longitude': [78.7800],
    'Order_Date': ['26-09-2025'],
    'Order_Time': ['15:30'],
    'Pickup_Time': ['15:35'], # This is not used by the model but kept for consistency
}


# Convert the dictionary to a pandas DataFrame
new_order_df = pd.DataFrame(new_order_data)

print("\nOriginal new order data:")
print(new_order_df)


print("\nMaking a prediction on the new data...")

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers.
    return c * r

new_order_df['Distance_km'] = haversine(
    new_order_df['Store_Latitude'], new_order_df['Store_Longitude'],
    new_order_df['Drop_Latitude'], new_order_df['Drop_Longitude']
)

# 2. Extract time-based features
order_datetime = pd.to_datetime(new_order_df['Order_Date'] + ' ' + new_order_df['Order_Time'], errors='coerce')
new_order_df['Order_Hour'] = order_datetime.dt.hour
new_order_df['Order_Day_of_Week'] = order_datetime.dt.dayofweek

# The pipeline knows which columns to use. We don't need to drop the original lat/lon/date cols,
# as the pipeline's preprocessor will only select the features it was trained on.

print("\nData after feature engineering (ready for prediction):")
print(new_order_df[['Agent_Age', 'Agent_Rating', 'Distance_km', 'Order_Hour', 'Traffic', 'Weather']].head())


# --- 3. Make a Prediction ---
print("\nMaking a prediction on the engineered data...")


# Now, the DataFrame has the 'Distance_km' column and the others the model expects.
predicted_time = loaded_pipeline.predict(new_order_df)

print(predicted_time)



predicted_time_in_minutes = predicted_time[0]
print(predicted_time_in_minutes)


# --- Prediction Result ---
print("\n--- Prediction Result ---")
hours = int(predicted_time_in_minutes / 60)
minutes = int(predicted_time_in_minutes % 60)
print(f"Predicted delivery time: {predicted_time_in_minutes:.2f} minutes")
print(f"(This is approximately {hours} hours and {minutes} minutes)")
print("------------------------")




