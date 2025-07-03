import sys
import os
import streamlit as st
import pandas as pd
from ads.analyzer import run_ad_analysis
from inventory.predictor import run_inventory_forecast

# Setup paths for uploaded data
UPLOADS_PATH = "data/uploads"
OUTPUTS_PATH = "data/outputs"

# Set up page
st.set_page_config(page_title="📊 Vibrant Analytics Dashboard", layout="wide")
st.title("📊 Vibrant Analytics Dashboard")
st.markdown("Welcome to **Vibrant Analytics Dashboard**, your tool to analyze and forecast inventory and ad campaigns.")

# Add a custom logo (optional)
st.image("https://your-logo-url.com/logo.png", width=200)  # Add your own logo URL here

# Sidebar Navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Choose a module", ["📈 Ad Campaign Analyzer", "🏬 Inventory Forecasting"])

# -----------------------
# Tab 1: Ad Campaign
# -----------------------
if tabs == "📈 Ad Campaign Analyzer":
    st.header("Ad Campaign Analyzer")
    st.markdown("Upload the following 4 reports to analyze the ad campaigns:")

    # Use columns for file upload
    col1, col2 = st.columns(2)
    with col1:
        adv = st.file_uploader("1. Advertised Report", type="xlsx")
        tgt = st.file_uploader("2. Targeting Report", type="xlsx")
    with col2:
        sales = st.file_uploader("3. Sales Report", type="xlsx")
        orders = st.file_uploader("4. Orders Report", type="xlsx")

    # Add button with enhanced design
    if st.button("▶️ Run Ad Analysis", help="Click to run the ad analysis on the uploaded reports"):
        if None in (adv, tgt, sales, orders):
            st.warning("⚠️ Please upload all 4 files.")
        else:
            # Save uploads
            os.makedirs(f"{UPLOADS_PATH}/ads", exist_ok=True)
            file_paths = {}
            for f, name in zip([adv, tgt, sales, orders], ['adv.xlsx', 'tgt.xlsx', 'sales.xlsx', 'orders.xlsx']):
                path = f"{UPLOADS_PATH}/ads/{name}"
                with open(path, "wb") as out: out.write(f.read())
                file_paths[name.split('.')[0]] = path

            # Run analysis
            with st.spinner("Running ad analysis..."):
                df_result = run_ad_analysis(
                    adv_path=file_paths['adv'],
                    tgt_path=file_paths['tgt'],
                    sales_path=file_paths['sales'],
                    orders_path=file_paths['orders']
                )
                os.makedirs(f"{OUTPUTS_PATH}/ads", exist_ok=True)
                out_path = f"{OUTPUTS_PATH}/ads/ASIN-Keyword-Final.xlsx"
                df_result.to_excel(out_path, index=False)

                st.success("✅ Analysis complete!")
                with open(out_path, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name="ASIN-Keyword-Final.xlsx")

# -----------------------
# Tab 2: Inventory Forecasting
# -----------------------
if tabs == "🏬 Inventory Forecasting":
    st.header("Inventory Forecasting")
    st.markdown("Upload the latest warehouse sales files (CSV) to forecast inventory demand:")

    # Use columns for file upload
    col1, col2, col3 = st.columns(3)
    with col1:
        csv1 = st.file_uploader("Sales File 1", type="csv")
    with col2:
        csv2 = st.file_uploader("Sales File 2", type="csv")
    with col3:
        csv3 = st.file_uploader("Sales File 3", type="csv")

    # Add button with enhanced design
    if st.button("▶️ Run Inventory Forecast", help="Click to forecast inventory demand based on the uploaded sales files"):
        if None in (csv1, csv2, csv3):
            st.warning("⚠️ Please upload all 3 sales files.")
        else:
            # Save uploads
            os.makedirs(f"{UPLOADS_PATH}/inventory", exist_ok=True)
            input_paths = []
            for i, f in enumerate([csv1, csv2, csv3]):
                path = f"{UPLOADS_PATH}/inventory/sales_{i+1}.csv"
                with open(path, "wb") as out: out.write(f.read())
                input_paths.append(path)

            with st.spinner("Running inventory forecast..."):
                df_forecast = run_inventory_forecast(input_paths)
                os.makedirs(f"{OUTPUTS_PATH}/inventory", exist_ok=True)
                out_path = f"{OUTPUTS_PATH}/inventory/Predicted_Weekly_Demand_Integrated.xlsx"
                df_forecast.to_excel(out_path, index=False)

                st.success("✅ Forecasting complete!")
                with open(out_path, "rb") as f:
                    st.download_button("📥 Download Forecast", f, file_name="Predicted_Weekly_Demand_Integrated.xlsx")
