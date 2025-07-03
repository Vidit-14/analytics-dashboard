import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime
from geopy.distance import geodesic
import os

def run_forecast_with_coords(file_paths: list, coord_file: str = "postal_code_coords.csv") -> pd.DataFrame:
    # === STEP 1: Load and Combine CSVs ===
    dataframes = [pd.read_csv(file) for file in file_paths]
    df = pd.concat(dataframes, ignore_index=True)
    df['Customer Shipment Date'] = pd.to_datetime(df['Customer Shipment Date'])
    df = df.rename(columns={'FC': 'Warehouse ID'})
    df = df[['Customer Shipment Date', 'ASIN', 'Quantity', 'Warehouse ID', 'Shipment To Postal Code']]
    df['Shipment To Postal Code'] = pd.to_numeric(df['Shipment To Postal Code'], errors='coerce')

    # === STEP 2: Load postal code coordinates ===
    coord_df = pd.read_csv(coord_file)
    coord_df = coord_df.dropna(subset=['Pincode', 'Latitude', 'Longitude'])
    postal_coords = coord_df.set_index('Pincode')[['Latitude', 'Longitude']].to_dict('index')

    # === STEP 3: Adjust FC if too far from shipping address ===
    def get_closest_fc(ship_postal, original_fc):
        try:
            ship_coord = postal_coords[int(ship_postal)]
        except:
            return original_fc
        try:
            original_coord = postal_coords[int(original_fc)]
        except:
            return original_fc
        dist = geodesic((ship_coord['Latitude'], ship_coord['Longitude']),
                        (original_coord['Latitude'], original_coord['Longitude'])).km
        if dist <= 300:
            return original_fc
        min_dist = float('inf')
        nearest_fc = original_fc
        for fc_pincode, coord in postal_coords.items():
            d = geodesic((ship_coord['Latitude'], ship_coord['Longitude']),
                         (coord['Latitude'], coord['Longitude'])).km
            if d < min_dist:
                min_dist = d
                nearest_fc = fc_pincode
        return nearest_fc

    df['Warehouse ID'] = df.apply(lambda row: get_closest_fc(row['Shipment To Postal Code'], row['Warehouse ID']), axis=1)

    # === STEP 4: Week Encoding ===
    df['Year'] = df['Customer Shipment Date'].dt.isocalendar().year
    df['Week No'] = df['Customer Shipment Date'].dt.isocalendar().week
    df = df[['Week No', 'Year', 'ASIN', 'Quantity', 'Warehouse ID']]
    df['week_index'] = df.groupby(['Year', 'Week No']).ngroup()
    max_index = df['week_index'].max()
    df = df[df['week_index'] >= max_index - 9]

    # === STEP 5: Label Encoding ===
    asin_encoder = LabelEncoder()
    wh_encoder = LabelEncoder()
    df['ASIN_Code'] = asin_encoder.fit_transform(df['ASIN'])
    df['Warehouse_Code'] = wh_encoder.fit_transform(df['Warehouse ID'])

    # === STEP 6: Lag Features ===
    df = df.sort_values(['ASIN', 'Warehouse ID', 'Year', 'Week No'])
    for lag in [1, 2, 3]:
        df[f'Lag_{lag}'] = df.groupby(['ASIN', 'Warehouse ID'])['Quantity'].shift(lag)
    df_model = df.dropna()

    # === STEP 7: Train XGBoost Model ===
    X = df_model[['ASIN_Code', 'Warehouse_Code', 'Lag_1', 'Lag_2', 'Lag_3']]
    y = df_model['Quantity']
    model = XGBRegressor()
    model.fit(X, y)
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # === STEP 8: Prepare Input for Prediction ===
    today = datetime.today()
    target_year, target_week = today.isocalendar().year, today.isocalendar().week + 1
    latest_weeks = df.groupby(['ASIN', 'Warehouse ID']).tail(3)
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

    # === STEP 9: Prediction ===
    if not input_df.empty:
        X_pred = input_df[['ASIN_Code', 'Warehouse_Code', 'Lag_1', 'Lag_2', 'Lag_3']]
        input_df['predicted demand'] = np.ceil(model.predict(X_pred)).astype(int)
        input_df['Year'] = target_year
        input_df['Week No'] = target_week
        output_df = input_df[['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand']]
    else:
        output_df = pd.DataFrame(columns=['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand'])

    # === STEP 10: Fallback 1-unit Prediction for Active ASIN-WH ===
    activity_df = df.groupby(['ASIN', 'Warehouse ID'])['Quantity'].sum().reset_index()
    activity_df = activity_df[activity_df['Quantity'] > 0]
    activity_df['Year'] = target_year
    activity_df['Week No'] = target_week
    activity_df['predicted demand'] = 1
    activity_df = activity_df[['ASIN', 'Warehouse ID', 'Year', 'Week No', 'predicted demand']]

    # === STEP 11: Merge and Return ===
    final_df = pd.concat([output_df, activity_df], ignore_index=True)
    final_df = final_df.groupby(['ASIN', 'Warehouse ID', 'Year', 'Week No'])['predicted demand'].max().reset_index()

    return final_df
