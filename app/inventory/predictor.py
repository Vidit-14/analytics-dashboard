import pandas as pd
import numpy as np
import geopy.distance
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from datetime import datetime
import pickle
import time
import os

def run_inventory_forecast(file_paths: list) -> pd.DataFrame:
    # === STEP 1: Load and Combine CSVs ===
    dataframes = [pd.read_csv(file, delimiter=',') for file in file_paths]
    df = pd.concat(dataframes, ignore_index=True)
    df['Customer Shipment Date'] = pd.to_datetime(df['Customer Shipment Date'])
    df = df.rename(columns={'FC': 'Warehouse ID', 'Shipment To Postal Code': 'Ship Postal Code'})
    df['Ship Postal Code'] = pd.to_numeric(df['Ship Postal Code'], errors='coerce')

    # === STEP 2: FC-to-Pincode Mapping ===
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

    # === STEP 3: Load or Generate Pincode Coordinates ===
    geolocator = Nominatim(user_agent="fc_distance_mapper")
    coord_cache_file = os.path.join(os.path.dirname(__file__), "postal_code_coords.csv")
    if os.path.exists(coord_cache_file):
        pin_to_coord_df = pd.read_csv(coord_cache_file)
        pin_to_coord = {int(row['Pincode']): (row['Latitude'], row['Longitude']) for _, row in pin_to_coord_df.iterrows()}
    else:
        pin_to_coord = {}

    def get_coords(pin):
        if pd.isna(pin): return None
        pin = int(pin)
        if pin in pin_to_coord: return pin_to_coord[pin]
        try:
            location = geolocator.geocode(f"India {pin}", timeout=10)
            if location:
                coords = (location.latitude, location.longitude)
                pin_to_coord[pin] = coords
                time.sleep(1)
                return coords
        except: pass
        pin_to_coord[pin] = None
        return None

    all_pins = set(df['Ship Postal Code'].dropna().astype(int).unique()) | set(fc_to_pincode.values())
    for pin in all_pins: get_coords(pin)

    # Save coordinate cache
    coord_df = pd.DataFrame([
        {'Pincode': pin, 'Latitude': coords[0], 'Longitude': coords[1]}
        for pin, coords in pin_to_coord.items() if coords is not None
    ])
    coord_df.to_csv(coord_cache_file, index=False)

    # === STEP 4: Classify Zone Type and Assign Closest FC ===
    def classify_and_assign(row):
        current_fc = row['Warehouse ID']
        ship_pin = row['Ship Postal Code']
        current_fc_pin = fc_to_pincode.get(current_fc)
        ship_coords = pin_to_coord.get(int(ship_pin)) if pd.notna(ship_pin) and int(ship_pin) in pin_to_coord else None
        current_coords = pin_to_coord.get(current_fc_pin)
        if ship_coords and current_coords:
            try:
                distance = geopy.distance.distance(current_coords, ship_coords).km
                if distance <= 300:
                    return pd.Series(['Regional', current_fc])
            except: pass
        min_dist = float('inf')
        closest_fc = current_fc
        for fc, fc_pin in fc_to_pincode.items():
            fc_coords = pin_to_coord.get(fc_pin)
            if fc_coords and ship_coords:
                try:
                    dist = geopy.distance.distance(fc_coords, ship_coords).km
                    if dist < min_dist:
                        min_dist = dist
                        closest_fc = fc
                except: continue
        return pd.Series(['Long Zone', closest_fc])

    df[['Zone Type', 'Assigned FC']] = df.apply(classify_and_assign, axis=1)
    df['Warehouse ID'] = df['Assigned FC']

    # === STEP 5A: Format Sales File (Not returned, just saved)
    df['Year'] = df['Customer Shipment Date'].dt.isocalendar().year
    df['Week No'] = df['Customer Shipment Date'].dt.isocalendar().week
    df['Quantity'] = 1
    sales_output = df[['ASIN', 'Amazon Order Id', 'Warehouse ID', 'Quantity', 'Zone Type']]
    sales_output.columns = ['ASIN', 'Order ID', 'Warehouse', 'Quantity', 'Long zone or Regional']
    sales_output.to_excel("Formatted_Sales_File.xlsx", index=False)

    # === STEP 5B: Forecasting with XGBoost ===
    sales_df = df[['Customer Shipment Date', 'ASIN', 'Quantity', 'Warehouse ID', 'Year', 'Week No']]
    sales_df['week_index'] = sales_df.groupby(['Year', 'Week No']).ngroup()
    max_index = sales_df['week_index'].max()
    sales_df = sales_df[sales_df['week_index'] >= max_index - 11]

    asin_encoder = LabelEncoder()
    wh_encoder = LabelEncoder()
    sales_df['ASIN_Code'] = asin_encoder.fit_transform(sales_df['ASIN'])
    sales_df['Warehouse_Code'] = wh_encoder.fit_transform(sales_df['Warehouse ID'])

    sales_df = sales_df.sort_values(['ASIN', 'Warehouse ID', 'Year', 'Week No'])
    for lag in [1, 2, 3]:
        sales_df[f'Lag_{lag}'] = sales_df.groupby(['ASIN', 'Warehouse ID'])['Quantity'].shift(lag)

    df_model = sales_df.dropna()
    X = df_model[['ASIN_Code', 'Warehouse_Code', 'Lag_1', 'Lag_2', 'Lag_3']]
    y = df_model['Quantity']
    model = XGBRegressor()
    model.fit(X, y)

    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    latest_weeks = sales_df.groupby(['ASIN', 'Warehouse ID']).tail(3)
    latest_weeks = latest_weeks.sort_values(['ASIN', 'Warehouse ID', 'Year', 'Week No'])
    input_rows = []
    for (asin, warehouse), group in latest_weeks.groupby(['ASIN', 'Warehouse ID']):
        if len(group) == 3:
            quantities = group['Quantity'].values
            input_rows.append({
                'ASIN': asin,
                'Warehouse ID': warehouse,
                'ASIN_Code': asin_encoder.transform([asin])[0],
                'Warehouse_Code': wh_encoder.transform([warehouse])[0],
                'Lag_1': quantities[2],
                'Lag_2': quantities[1],
                'Lag_3': quantities[0]
            })

    input_df = pd.DataFrame(input_rows)
    today = datetime.today()
    target_year, target_week = today.isocalendar().year, today.isocalendar().week + 1

    if not input_df.empty:
        X_pred = input_df[['ASIN_Code', 'Warehouse_Code', 'Lag_1', 'Lag_2', 'Lag_3']]
        input_df['predicted demand'] = np.ceil(model.predict(X_pred)).astype(int)
        input_df['Year'] = target_year
        input_df['Week No'] = target_week
        output_df = input_df[['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand']]
    else:
        output_df = pd.DataFrame(columns=['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand'])

    activity_df = sales_df.groupby(['ASIN', 'Warehouse ID'])['Quantity'].sum().reset_index()
    activity_df = activity_df[activity_df['Quantity'] > 0]
    activity_df['Year'] = target_year
    activity_df['Week No'] = target_week
    activity_df['predicted demand'] = 1
    activity_df = activity_df[['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand']]

    final_df = pd.concat([output_df, activity_df], ignore_index=True)
    final_df = final_df.groupby(['ASIN', 'Warehouse ID', 'Year', 'Week No'])['predicted demand'].max().reset_index()

    return final_df
