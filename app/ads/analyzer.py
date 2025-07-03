import pandas as pd
import numpy as np

def run_ad_analysis(adv_path, tgt_path, sales_path, orders_path):
    # --------------------------
    # 1. Load Input Files
    # --------------------------
    adv = pd.read_excel(adv_path)
    tgt = pd.read_excel(tgt_path)
    sales = pd.read_excel(sales_path)
    orders = pd.read_excel(orders_path)

    # --------------------------
    # 2. ASIN-WISE PROFIT MARGINS
    # --------------------------
    sales.columns = sales.columns.str.strip().str.lower()
    orders.columns = orders.columns.str.strip().str.lower()

    sales_df = sales[["date/time", "order id", "product sales", "total"]].copy()
    orders_df = orders[["amazon order id", "asin"]].copy()
    sales_df.columns = ["order_date", "order_id", "product_sales", "total"]
    orders_df.columns = ["order_id", "asin"]

    merged_sales = pd.merge(sales_df, orders_df, on="order_id", how="inner")
    merged_sales["product_sales"] = merged_sales["product_sales"].astype(str).str.replace(",", "", regex=False).astype(float)
    merged_sales["total"] = merged_sales["total"].astype(str).str.replace(",", "", regex=False).astype(float)
    merged_sales = merged_sales[merged_sales["product_sales"] != 0]
    merged_sales["profit_margin_%"] = (merged_sales["total"] / merged_sales["product_sales"]) * 100
    merged_sales["profit_margin_%"] = merged_sales["profit_margin_%"].round(2)
    merged_sales["product_sales"] = merged_sales["product_sales"].abs()
    merged_sales["total"] = merged_sales["total"].abs()

    asin_margin = merged_sales.groupby("asin")["profit_margin_%"].mean().reset_index()
    asin_margin["breakeven_acos"] = (1 / (asin_margin["profit_margin_%"] / 100)).round(2)

    # --------------------------
    # 3. Clean Ads Data
    # --------------------------
    adv.columns = adv.columns.str.strip()
    tgt.columns = tgt.columns.str.strip()

    adv_keep = [
        'Campaign Name','Ad Group Name','Country','Advertised ASIN',
        'Impressions','Clicks','Click-Thru Rate (CTR)','Cost Per Click (CPC)',
        'Spend','14 Day Total Sales','Total Advertising Cost of Sales (ACOS)',
        'Total Return on Advertising Spend (ROAS)',
        '14 Day Total Orders (#)','14 Day Total Units (#)','14 Day Conversion Rate'
    ]
    tgt_keep = [
        'Campaign Name','Ad Group Name','Targeting','Match Type',
        'Impressions','Top-of-search Impression Share','Clicks',
        'Click-Thru Rate (CTR)','Cost Per Click (CPC)','Spend',
        'Total Advertising Cost of Sales (ACOS)',
        'Total Return on Advertising Spend (ROAS)',
        '14 Day Total Sales','14 Day Total Orders (#)','14 Day Total Units (#)',
        '14 Day Conversion Rate'
    ]
    adv = adv[adv_keep]
    tgt = tgt[tgt_keep]

    adv = adv.rename(columns={
        'Advertised ASIN':'asin',
        'Impressions':'impr_asin','Clicks':'clicks_asin',
        'Click-Thru Rate (CTR)':'ctr_asin','Cost Per Click (CPC)':'cpc_asin',
        'Spend':'spend_asin','14 Day Total Sales':'sales_asin',
        'Total Advertising Cost of Sales (ACOS)':'acos_asin',
        'Total Return on Advertising Spend (ROAS)':'roas_asin',
        '14 Day Total Orders (#)':'orders_asin','14 Day Total Units (#)':'units_asin',
        '14 Day Conversion Rate':'cvr_asin'
    })
    tgt = tgt.rename(columns={
        'Impressions':'impr_kw','Clicks':'clicks_kw',
        'Click-Thru Rate (CTR)':'ctr_kw','Cost Per Click (CPC)':'cpc_kw',
        'Spend':'spend_kw','14 Day Total Sales':'sales_kw',
        'Total Advertising Cost of Sales (ACOS)':'acos_kw',
        'Total Return on Advertising Spend (ROAS)':'roas_kw',
        '14 Day Total Orders (#)':'orders_kw','14 Day Total Units (#)':'units_kw',
        '14 Day Conversion Rate':'cvr_kw',
        'Top-of-search Impression Share':'toss_kw'
    })

    for col in ['Campaign Name','Ad Group Name']:
        adv[col] = adv[col].astype(str).str.strip()
        tgt[col] = tgt[col].astype(str).str.strip()

    merged = adv.merge(tgt, on=["Campaign Name","Ad Group Name"], how="left")

    # Normalize ASIN format before merging
    merged["asin"] = merged["asin"].astype(str).str.strip().str.upper()
    asin_margin["asin"] = asin_margin["asin"].astype(str).str.strip().str.upper()

    merged = merged.merge(asin_margin, on="asin", how="left")
    merged["breakeven_acos"] = merged["breakeven_acos"].fillna(0.40)

    num_cols = [c for c in merged.columns if c.endswith(('_asin','_kw'))]
    merged[num_cols] = merged[num_cols].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0))

    # --------------------------
    # 4. Diagnostics
    # --------------------------
    IMP_MIN = 100
    CTR_MIN = 0.003
    CVR_MIN = 0.05

    def rate_row(spend, sales, acos, ctr, cvr, impr, break_acos):
        reasons = []
        if spend > 0 and sales == 0:
            reasons.append('Spend>0 & No Sales')
        if acos > break_acos:
            reasons.append('High ACOS')
        if ctr < CTR_MIN:
            reasons.append('Low CTR')
        if cvr < CVR_MIN:
            reasons.append('Low CVR')
        if impr < IMP_MIN:
            reasons.append('Low Impressions')

        if not reasons:
            return 'Healthy', ''
        if 'Spend>0 & No Sales' in reasons:
            status = 'Bleeding'
        elif 'High ACOS' in reasons:
            status = 'Unprofitable'
        else:
            status = 'Needs Improvement'
        return status, '; '.join(reasons)

    def keyword_suggestion(reason):
        if 'Spend>0 & No Sales' in reason or 'High ACOS' in reason:
            return 'Decrease bid or add negatives'
        elif 'Low CTR' in reason:
            return 'Improve targeting or ad copy'
        elif 'Low CVR' in reason:
            return 'Improve landing page/relevancy'
        elif 'Low Impressions' in reason:
            return 'Increase bid'
        else:
            return 'Monitor'

    def asin_suggestion(reason):
        if 'Spend>0 & No Sales' in reason or 'High ACOS' in reason:
            return 'Lower bid / improve listing / add negatives'
        elif 'Low CTR' in reason:
            return 'Improve main image/title'
        elif 'Low CVR' in reason:
            return 'Check product page & reviews'
        elif 'Low Impressions' in reason:
            return 'Raise bid or broaden targeting'
        else:
            return 'Monitor'

    perf_asin, reason_asin = zip(*merged.apply(
        lambda r: rate_row(r.spend_asin, r.sales_asin, r.acos_asin,
                           r.ctr_asin, r.cvr_asin, r.impr_asin, r.breakeven_acos), axis=1))
    perf_kw, reason_kw = zip(*merged.apply(
        lambda r: rate_row(r.spend_kw, r.sales_kw, r.acos_kw,
                           r.ctr_kw, r.cvr_kw, r.impr_kw, r.breakeven_acos), axis=1))

    merged['Performance_asin'] = perf_asin
    merged['Reason_asin'] = reason_asin
    merged['Suggestion_asin'] = [asin_suggestion(r) for r in reason_asin]
    merged['Performance_kw'] = perf_kw
    merged['Reason_kw'] = reason_kw
    merged['Suggestion_kw'] = [keyword_suggestion(r) for r in reason_kw]

    # --------------------------
    # 5. Neat Column Placement
    # --------------------------
    def insert_after(df, after_col, new_cols):
        base_idx = df.columns.get_loc(after_col) + 1
        for i, col in enumerate(new_cols):
            df.insert(base_idx + i, col, df.pop(col))

    insert_after(merged, 'cvr_asin', ['profit_margin_%', 'breakeven_acos'])
    insert_after(merged, 'breakeven_acos', ['Performance_asin','Reason_asin','Suggestion_asin'])
    insert_after(merged, 'cvr_kw', ['Performance_kw','Reason_kw','Suggestion_kw'])

    # --------------------------
    # 6. Return Final DataFrame
    # --------------------------
    return merged
