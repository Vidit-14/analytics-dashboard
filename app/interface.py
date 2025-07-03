import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from ads.analyzer import run_ad_analysis
from inventory.predictor import run_inventory_forecast

UPLOADS_PATH = "data/uploads"
OUTPUTS_PATH = "data/outputs"

# Set up page
st.set_page_config(page_title="Analytics Dashboard", layout="wide")
st.title("📊 Vibrant Analytics Dashboard")

# Create sub-tabs for each module
tabs = st.tabs(["📈 Ad Campaign Analyzer", "🏬 Inventory Forecasting"])

# -----------------------
# Tab 1: Ad Campaign
# -----------------------
with tabs[0]:
    st.header("Ad Campaign Analyzer")
    st.markdown("Upload 4 reports: Advertised, Targeting, Sales, and Orders.")

    adv = st.file_uploader("1. Advertised Report", type="xlsx", key="adv")
    tgt = st.file_uploader("2. Targeting Report", type="xlsx", key="tgt")
    sales = st.file_uploader("3. Sales Report", type="xlsx", key="sales")
    orders = st.file_uploader("4. Orders Report", type="xlsx", key="orders")

    if st.button("▶️ Run Ad Analysis"):
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
with tabs[1]:
    st.header("Inventory Forecasting")
    st.markdown("Upload the 3 latest warehouse sales files (CSV).")

    csv1 = st.file_uploader("Sales File 1", type="csv", key="csv1")
    csv2 = st.file_uploader("Sales File 2", type="csv", key="csv2")
    csv3 = st.file_uploader("Sales File 3", type="csv", key="csv3")

    if st.button("▶️ Run Inventory Forecast"):
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
