import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle
import time
from concurrent.futures import ThreadPoolExecutor


def run_inventory_forecast(sales_file_path: str, warehouse_file_path: str, historical_data_path: str) -> pd.DataFrame:
    """
    Run inventory forecast with moving window of the most recent 12 weeks of data.

    Args:
    - sales_file_path (str): Path to the uploaded sales data file (CSV).
    - warehouse_file_path (str): Path to the uploaded warehouse balance file (CSV).
    - historical_data_path (str): Path to the historical sales data file (CSV).
    
    Returns:
    - pd.DataFrame: Forecasted demand for the next week.
    """

    # === STEP 1: Load New Sales Data ===
    new_sales_df = pd.read_csv(sales_file_path)
    new_sales_df['Customer Shipment Date'] = pd.to_datetime(new_sales_df['Customer Shipment Date'])
    new_sales_df = new_sales_df.rename(columns={'FC': 'Warehouse ID', 'Shipment To Postal Code': 'Ship Postal Code'})
    new_sales_df['Ship Postal Code'] = pd.to_numeric(new_sales_df['Ship Postal Code'], errors='coerce')

    # === STEP 2: Load Warehouse Balance Data ===
    warehouse_balance_df = pd.read_csv(warehouse_file_path)

    # Rename 'Location' to 'Warehouse ID' since that's the column name in the uploaded file
    if 'Location' in warehouse_balance_df.columns:
        warehouse_balance_df.rename(columns={'Location': 'Warehouse ID'}, inplace=True)
    else:
        raise ValueError("The 'Location' column is missing in the warehouse balance file.")

    # === STEP 3: Load Historical Data (March, April, May) ===
    if os.path.exists(historical_data_path):
        historical_data_df = pd.read_csv(historical_data_path)
    else:
        historical_data_df = pd.DataFrame()  # Empty dataframe if no historical data exists

    # === STEP 4: Combine New Data with Historical Data ===
    combined_df = pd.concat([historical_data_df, new_sales_df], ignore_index=True)
    combined_df = combined_df.sort_values('Customer Shipment Date')

    # === STEP 5: Ensure 'Customer Shipment Date' is in datetime format ===
    combined_df['Customer Shipment Date'] = pd.to_datetime(combined_df['Customer Shipment Date'], errors='coerce')

    # === STEP 6: Implement Moving Window (12 Weeks of Data) ===
    today = datetime.today()
    cutoff_date = today - timedelta(weeks=12)
    combined_df = combined_df[combined_df['Customer Shipment Date'] >= cutoff_date]

    # === STEP 7: Update Historical Data with Combined Data ===
    combined_df.to_csv(historical_data_path, index=False)

    # === STEP 8: Define FC-to-Pincode Mapping ===
    fc_to_pincode = {
        'SGAA': 781132, 'SGAC': 781101, 'SPAB': 801103, 'DEX3': 110044, 'PNQ2': 110044, 'DEX8': 110044,
        'AMD2': 382220, 'SAME': 387570, 'DEL2': 122105, 'DEL4': 122503, 'DEL5': 122413, 'DED3': 122506,
        'DED5': 122103, 'SDEG': 122105, 'SDEB': 124108, 'XNRW': 124108, 'BLR4': 562149, 'BLR5': 562114,
        'BLR7': 562107, 'BLR8': 562149, 'BLX1': 563160, 'XSAJ': 562132, 'BOM5': 421302, 'BOM7': 421302,
        'ISK3': 421302, 'PNQ3': 410501, 'SBOB': 421101, 'XWAA': 421302, 'SBHF': 462030, 'SIDA': 453771,
        'ATX1': 141113, 'SATB': 141113, 'JPX1': 302016, 'JPX2': 303007, 'HYD8': 500108, 'HYD3': 500108, 'SHYH': 500108,
        'SHYB': 502279, 'XSAD': 502279, 'XSIP': 502279, 'MAA4': 601206, 'CJB1': 641201, 'SMAB': 601103,
        'SCJF': 641402, 'XSIR': 601103, 'LKO1': 226401, 'SLKD': 226401, 'CCX1': 711322, 'CCX2': 711302,
        'SCCE': 711313, 'XECP': 711401, 'PAX1': 800009
    }

    # === STEP 9: Load or Generate Pincode Coordinates Mapping ===
    geolocator = Nominatim(user_agent="fc_distance_mapper")
    coord_cache_file = 'postal_code_coords.csv'

    # Check if we already have a cached coordinates file
    if os.path.exists(coord_cache_file):
        pin_to_coord_df = pd.read_csv(coord_cache_file)
        pin_to_coord = {int(row['Pincode']): (row['Latitude'], row['Longitude']) for _, row in pin_to_coord_df.iterrows()}
    else:
        pin_to_coord = {}

    # === STEP 10: Geocode Function with Caching ===
    def geocode_postal(pin):
        """Geocode postal code using geopy."""
        if pd.isna(pin) or pin in pin_to_coord:
            return pin_to_coord.get(pin, None)

        try:
            location = geolocator.geocode(f"India {int(pin)}", timeout=10)
            if location:
                coords = (location.latitude, location.longitude)
                pin_to_coord[pin] = coords
                return coords
        except Exception as e:
            print(f"Error geocoding postal code {pin}: {e}")

        return None

    def fetch_coordinates(postal_codes):
        """Batch process postal codes using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(geocode_postal, postal_codes))
        return results

    # === STEP 11: Pre-fetch Coordinates for All Unique Postal Codes ===
    all_pins = set(combined_df['Ship Postal Code'].dropna().astype(int).unique()) | set(fc_to_pincode.values())
    coordinates = fetch_coordinates(all_pins)

    # Cache coordinates to CSV
    coord_df = pd.DataFrame([
        {'Pincode': pin, 'Latitude': coords[0], 'Longitude': coords[1]}
        for pin, coords in zip(all_pins, coordinates) if coords is not None
    ])
    coord_df.to_csv(coord_cache_file, index=False)
    print(f"📍 Saved {len(coord_df)} pincode coordinates to '{coord_cache_file}'")

    # === STEP 12: Adjust FC if Too Far From Shipping Address ===
    def get_closest_fc(ship_postal, original_fc):
        """Find the closest FC based on distance from the shipping postal code."""
        try:
            ship_coord = pin_to_coord[int(ship_postal)]
        except:
            return original_fc  # return original FC if no coordinates available

        try:
            original_fc_pin = fc_to_pincode.get(original_fc)
            original_coord = pin_to_coord.get(original_fc_pin)
            if original_coord:
                dist = geodesic((ship_coord[0], ship_coord[1]), (original_coord[0], original_coord[1])).km
                if dist <= 300:
                    return original_fc
        except:
            pass

        # Find the nearest FC if distance > 300 km
        min_dist = float('inf')
        nearest_fc = original_fc
        for fc, fc_pin in fc_to_pincode.items():
            fc_coords = pin_to_coord.get(fc_pin)
            if fc_coords and ship_coord:
                dist = geodesic((ship_coord[0], ship_coord[1]), (fc_coords[0], fc_coords[1])).km
                if dist < min_dist:
                    min_dist = dist
                    nearest_fc = fc
        return nearest_fc

    combined_df['Warehouse ID'] = combined_df.apply(lambda row: get_closest_fc(row['Ship Postal Code'], row['Warehouse ID']), axis=1)

    # === STEP 13: Prepare Data for ML Model ===
    combined_df['Year'] = combined_df['Customer Shipment Date'].dt.isocalendar().year
    combined_df['Week No'] = combined_df['Customer Shipment Date'].dt.isocalendar().week

    asin_encoder = LabelEncoder()
    wh_encoder = LabelEncoder()
    combined_df['ASIN_Code'] = asin_encoder.fit_transform(combined_df['ASIN'])
    combined_df['Warehouse_Code'] = wh_encoder.fit_transform(combined_df['Warehouse ID'])

    combined_df = combined_df.sort_values(['ASIN', 'Warehouse ID', 'Year', 'Week No'])
    for lag in [1, 2, 3]:
        combined_df[f'Lag_{lag}'] = combined_df.groupby(['ASIN', 'Warehouse ID'])['Quantity'].shift(lag)

    df_model = combined_df.dropna()
    X = df_model[['ASIN_Code', 'Warehouse_Code', 'Lag_1', 'Lag_2', 'Lag_3']]
    y = df_model['Quantity']

    model = XGBRegressor()
    model.fit(X, y)

    # === STEP 14: Make Predictions ===
    latest_weeks = combined_df.groupby(['ASIN', 'Warehouse ID']).tail(3)
    input_rows = []
    for (asin, wh), group in latest_weeks.groupby(['ASIN', 'Warehouse ID']):
        if len(group) == 3:
            lags = group['Quantity'].values[::-1]
            input_rows.append({
                'ASIN': asin,
                'Warehouse ID': wh,
                'ASIN_Code': asin_encoder.transform([asin])[0],
                'Warehouse_Code': wh_encoder.transform([wh])[0],
                'Lag_1': lags[0],
                'Lag_2': lags[1],
                'Lag_3': lags[2]
            })

    input_df = pd.DataFrame(input_rows)
    today = datetime.today()
    target_year, target_week = today.isocalendar().year, today.isocalendar().week + 1

    if not input_df.empty:
        X_pred = input_df[['ASIN_Code', 'Warehouse_Code', 'Lag_1', 'Lag_2', 'Lag_3']]
        input_df['predicted demand'] = np.ceil(model.predict(X_pred)).astype(int)
        input_df['Year'] = target_year
        input_df['Week No'] = target_week
        forecast_df = input_df[['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand']]
    else:
        forecast_df = pd.DataFrame(columns=['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand'])

    # === STEP 15: Fallback Demand = 1 for Active ASINs ===
    activity_df = combined_df.groupby(['ASIN', 'Warehouse ID'])['Quantity'].sum().reset_index()
    activity_df = activity_df[activity_df['Quantity'] > 0]
    activity_df['Year'] = target_year
    activity_df['Week No'] = target_week
    activity_df['predicted demand'] = 1
    fallback_df = activity_df[['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand']]

    # === STEP 16: Merge Demand ===
    final_demand = pd.concat([forecast_df, fallback_df], ignore_index=True)
    final_demand = final_demand.groupby(['ASIN', 'Warehouse ID', 'Year', 'Week No'])['predicted demand'].max().reset_index()

    return final_demand
