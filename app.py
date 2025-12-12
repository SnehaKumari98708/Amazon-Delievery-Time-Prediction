from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import warnings

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom function to handle sklearn compatibility issues
def load_model_with_fix(filename):
    try:
        # First try normal loading
        return joblib.load(filename)
    except AttributeError as e:
        if '_RemainderColsList' in str(e):
            print("Fixing sklearn compatibility issue...")
            # Create a mock class to handle the missing attribute
            import sklearn.compose._column_transformer
            class _RemainderColsList(list):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
            return joblib.load(filename)
        else:
            raise e

# Load the model pipeline with better error handling
try:
    pipeline_filename = 'delivery_time_prediction_pipeline.joblib'
    
    # Check if file exists
    if not os.path.exists(pipeline_filename):
        raise FileNotFoundError(f"Model file '{pipeline_filename}' not found.")
    
    # Try to load the model with compatibility fix
    loaded_pipeline = load_model_with_fix(pipeline_filename)
    print("Model loaded successfully!")
    model_loaded = True
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    loaded_pipeline = None
    model_loaded = False
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying alternative loading method...")
    
    # Alternative method using custom unpickler
    try:
        from sklearn.externals import joblib as alt_joblib
        loaded_pipeline = alt_joblib.load(pipeline_filename)
        print("Model loaded successfully with alternative method!")
        model_loaded = True
    except:
        print("All loading methods failed.")
        loaded_pipeline = None
        model_loaded = False

# Haversine function to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded or loaded_pipeline is None:
        return jsonify({
            'error': 'Model not loaded properly.',
            'solution': 'Please check sklearn version compatibility. Try: pip install scikit-learn==1.6.1'
        })
    
    try:
        # Get form data
        data = request.json
        
        # Create DataFrame from form data
        new_order_data = {
            'Agent_Age': [int(data['agent_age'])],
            'Agent_Rating': [int(data['agent_rating'])],
            'Weather': [data['weather']],
            'Traffic': [data['traffic']],
            'Vehicle': [data['vehicle']],
            'Area': [data['area']],
            'Category': [data['category']],
            'Store_Latitude': [float(data['store_lat'])],
            'Store_Longitude': [float(data['store_lng'])],
            'Drop_Latitude': [float(data['drop_lat'])],
            'Drop_Longitude': [float(data['drop_lng'])],
            'Order_Date': [data['order_date']],
            'Order_Time': [data['order_time']],
            'Pickup_Time': [data['pickup_time']],
        }
        
        # Convert to DataFrame
        new_order_df = pd.DataFrame(new_order_data)
        
        # Feature engineering
        new_order_df['Distance_km'] = haversine(
            new_order_df['Store_Latitude'], new_order_df['Store_Longitude'],
            new_order_df['Drop_Latitude'], new_order_df['Drop_Longitude']
        )
        
        # Extract time-based features
        order_datetime = pd.to_datetime(
            new_order_df['Order_Date'] + ' ' + new_order_df['Order_Time'], 
            errors='coerce'
        )
        new_order_df['Order_Hour'] = order_datetime.dt.hour
        new_order_df['Order_Day_of_Week'] = order_datetime.dt.dayofweek
        
        # Make prediction
        predicted_time = loaded_pipeline.predict(new_order_df)
        predicted_time_in_minutes = predicted_time[0]
        
        # Format the result
        hours = int(predicted_time_in_minutes / 60)
        minutes = int(predicted_time_in_minutes % 60)
        
        return jsonify({
            'success': True,
            'predicted_minutes': round(predicted_time_in_minutes, 2),
            'formatted_time': f"{hours} hours and {minutes} minutes",
            'distance_km': round(new_order_df['Distance_km'].iloc[0], 2)
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok' if model_loaded else 'error',
        'model_loaded': model_loaded,
        'sklearn_version': 'Unknown' if not model_loaded else 'Loaded successfully'
    })

if __name__ == '__main__':
    app.run(debug=True)