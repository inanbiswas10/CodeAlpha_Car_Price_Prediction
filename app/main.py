import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Append project root directory to python path

sys.path.append (os.path.abspath (os.path.join (os.path.dirname (__file__),"..")))
from src.data_loader import load_and_clean_car_data
from src.features import engineer_car_features

# ==========================================
# 1) PAGE CONFIGURATION & ENTERPRISE STYLING
# ==========================================

st.set_page_config (
    page_title = "AutoValuate AI",
    page_icon = "🚗",
    layout = "wide",
    initial_sidebar_state = "expanded",)

# High-Density Responsive Styling

st.markdown (
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #161B22 0%, #1F242D 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #8B949E;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #00D26A;
        font-size: 22px;
        font-weight: 800;
        margin-top: 4px;
    }
    .badge-deal {
        background-color: #238636;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 12px;
    }
    .badge-overprice {
        background-color: #DA3633;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 12px;
    }
    .window-header {
        background: #161B22;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 5px solid #00D26A;
        margin-bottom: 18px;
    }
    </style>
""",
    unsafe_allow_html = True,)

# ==========================================
# 2) DATA & MODEL LOADING PIPELINE
# ==========================================

@st.cache_resource
def load_trained_model ():
    model_path = os.path.join ("models","car_price_model.pkl")
    if not os.path.exists (model_path):
        st.error (
            "Model pipeline not found !! Please run 'python src/train.py' first")
        st.stop ()
    return joblib.load (model_path)

@st.cache_data
def load_dataset ():
    data_path = os.path.join ("data","car_data.csv")
    df_raw = load_and_clean_car_data (data_path)
    return engineer_car_features (df_raw)

pipeline = load_trained_model ()
df = load_dataset ()

# ==========================================
# 3) GLOBAL SESSION STATE INITIALIZATION
# ==========================================

if "predicted_price" not in st.session_state:
    st.session_state ["predicted_price"] = 5.25
    st.session_state ["present_price"] = 9.54
    st.session_state ["manufacture_year"] = 2018
    st.session_state ["kms_driven"] = 43000
    st.session_state ["fuel_type"] = "Petrol"
    st.session_state ["transmission"] = "Manual"
    st.session_state ["brand_name"] = "Swift"
    st.session_state ["seller_type"] = "Dealer"
    st.session_state ["owner"] = 0

# Header Banner

st.title ("AutoValuate AI - Used Car AI Valuation & Intelligence Platform")
st.caption (
    "CodeAlpha Portfolio Flagship | High-Density Multi-Window Analytical Dashboard")
st.divider ()

# ==========================================
# 4) NAVIGATION WINDOW SELECTOR
# ==========================================

st.sidebar.header ("🕹️ Section Management")
window_choice = st.sidebar.radio ("Switch Active Section",
    [
        "🎛️ Executive Command Center",
        "⚡ What-If Sensitivity Lab",
        "⚔️ Side By Side Comparison",
        "📊 Buy vs Lease Decision Matrix",
        "🌐 Regional RTO & Inflation",
        "🤝 Negotiation & TCO Matrix",
        "📈 Market Position Benchmark",
        "🔬 Visual Analytics Lab",
    ],
)

# ==========================================
# SECTION-1: EXECUTIVE COMMAND CENTER
# ==========================================

if window_choice == "🎛️ Executive Command Center":
    st.markdown (
        "<div class='window-header'><h3>🎛️ Executive Valuation Command Center</h3></div>",
        unsafe_allow_html = True,)

    # ----------------------------------------------------
    # TOP ROW: Vehicle Inputs (Left) | KPIs & Radar (Right)
    # ----------------------------------------------------

    top_left,top_right = st.columns ([1.0,1.8],gap = "large")

    with top_left:
        st.subheader ("🛠️ Vehicle Inputs")
        brand_name = st.text_input ("Car Model / Brand Name",value = "Swift")
        manufacture_year = st.slider ("Manufacturing Year",2000,2026,2018)
        present_price = st.number_input (
            "Original Showroom Price (Lakhs ₹)",0.5,100.0,9.54,0.1)
        kms_driven = st.number_input (
            "Odometer Reading (Kilometers)",100,500000,43000,1000)
        fuel_type = st.selectbox ("Fuel Type",["Petrol","Diesel","CNG"])
        seller_type = st.selectbox ("Seller Type",["Dealer","Individual"])
        transmission = st.selectbox ("Transmission Mode",["Manual","Automatic"])
        owner = st.selectbox ("Previous Owners",[0,1,2,3])

        luxury_brands = [
            "mercedes",
            "bmw",
            "audi",
            "land",
            "jaguar",
            "porsche",
            "volvo",
            "fortuner",
        ]
        is_luxury = (
            1 if any (b in brand_name.lower () for b in luxury_brands) else 0
        )

        current_year = 2026
        vehicle_age = (current_year-manufacture_year)
        kms_per_year = (
            kms_driven / vehicle_age if vehicle_age > 0 else kms_driven
        )

        input_data = pd.DataFrame (
            [
                {
                    "year": manufacture_year,
                    "present_price": present_price,
                    "kms_driven": kms_driven,
                    "fuel_type": fuel_type,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                    "vehicle_age": vehicle_age,
                    "is_luxury_brand": is_luxury,
                    "kms_per_year": kms_per_year,
                }
            ]
        )

        predicted_price = max (0.1,float (pipeline.predict (input_data)[0]))

        st.session_state ["predicted_price"] = predicted_price
        st.session_state ["present_price"] = present_price
        st.session_state ["manufacture_year"] = manufacture_year
        st.session_state ["kms_driven"] = kms_driven
        st.session_state ["fuel_type"] = fuel_type
        st.session_state ["transmission"] = transmission
        st.session_state ["brand_name"] = brand_name
        st.session_state ["seller_type"] = seller_type
        st.session_state ["owner"] = owner

    with top_right:
        st.subheader ("📊 Real Time Valuation Indicators")
        k1,k2,k3 = st.columns (3)
        k1.markdown (
            f'<div class="metric-card"><div class="metric-title">Predicted Resale</div><div class="metric-value">₹ {predicted_price:.2f} L</div></div>',
            unsafe_allow_html = True,)
        k2.markdown (
            f'<div class="metric-card"><div class="metric-title">Retained Value</div><div class="metric-value">{(predicted_price/present_price)*100:.1f} %</div></div>',
            unsafe_allow_html = True,)
        k3.markdown (
            f'<div class="metric-card"><div class="metric-title">Depreciation</div><div class="metric-value">₹ {(present_price - predicted_price):.2f} L</div></div>',
            unsafe_allow_html = True,)

        st.markdown ("<div style='margin-bottom: 20px;'></div>",unsafe_allow_html = True)

        st.subheader ("🕸️ 5-Axis Car Health Radar Diagnostic")
        ret_score = min (100,(predicted_price/present_price)*100)
        age_score = max (0,100-(vehicle_age*8))
        use_score = max (0,100-(kms_per_year/300))
        brand_score = 90 if is_luxury else 65
        liquidity_score = 85 if transmission == "Manual" else 75

        radar_df = pd.DataFrame (
            {
                "r": [
                    ret_score,
                    age_score,
                    use_score,
                    brand_score,
                    liquidity_score,
                ],
                "theta": [
                    "Retention",
                    "Age Health",
                    "Usage Rate",
                    "Brand Equity",
                    "Liquidity",
                ],
            }
        )

        fig_radar = px.line_polar (
            radar_df,
            r = "r",
            theta = "theta",
            line_close = True,
            template = "plotly_dark",
            height = 340,
        )
        fig_radar.update_traces (
            fill = "toself",
            fillcolor = "rgba(0, 210, 106, 0.25)",
            line_color = "#00D26A",
        )
        fig_radar.update_layout (
            margin = dict(l = 40,r = 40,t = 30,b = 30),  # Generous margins to prevent label clipping
            polar = dict(radialaxis = dict(visible = True,showticklabels = True)),
        )
        st.plotly_chart (fig_radar,use_container_width = True)

    st.markdown ("---")

    # ----------------------------------------------------
    # BOTTOM ROW: Depreciation Trajectory (Left) | Thresholds & Export (Right)
    # ----------------------------------------------------

    bot_left,bot_right = st.columns ([1.8,1.0],gap = "large")

    with bot_left:
        st.subheader ("📉 5-Year Depreciation Trajectory (Graphical Analysis)")
        future_years = np.array ([0,1,2,3,4,5])
        projected_prices = []
        for y in future_years:
            future_age = vehicle_age + y
            future_kms = kms_driven + (y * kms_per_year)
            future_input = input_data.copy ()
            future_input ["vehicle_age"] = future_age
            future_input ["kms_driven"] = future_kms
            future_input ["year"] = manufacture_year
            val = float (pipeline.predict (future_input)[0])
            projected_prices.append (max (0.05,val))

        dep_df = pd.DataFrame (
            {
                "Years from Now": future_years,
                "Estimated Value (Lakhs ₹)": projected_prices,
            }
        )
        fig_dep = px.line (
            dep_df,
            x = "Years from Now",
            y = "Estimated Value (Lakhs ₹)",
            markers = True,
            template = "plotly_dark",
            height = 320,
        )
        fig_dep.update_layout (margin = dict(l = 20,r = 20,t = 20,b = 20))
        st.plotly_chart (fig_dep,use_container_width = True)

    with bot_right:
        st.subheader ("💰 Recommended Price Thresholds")
        st.markdown ("<div style='margin-bottom: 15px;'></div>",unsafe_allow_html = True)
        
        b1,b2 = st.columns (2)
        b1.markdown (
            f"**Great Deal Target:**<br><span class='badge-deal' style='font-size: 15px;'>₹ {predicted_price*0.92:.2f} L</span>",
            unsafe_allow_html = True,)
        b2.markdown (
            f"**Overpriced Ceiling:**<br><span class='badge-overprice' style='font-size: 15px;'>₹ {predicted_price*1.08:.2f} L</span>",
            unsafe_allow_html = True,)

        st.markdown ("<div style='margin-bottom: 25px;'></div>",unsafe_allow_html = True)
        st.subheader ("📄 Valuation Export")
        report_data = f"""VEHICLE VALUATION CERTIFICATE\nModel: {brand_name} | Year: {manufacture_year}\nPredicted Market Value: ₹ {predicted_price:.2f} Lakhs"""
        st.download_button(
            label = "📥 Download Official Certificate (.txt)",
            data = report_data,
            file_name = f"{brand_name}_Valuation.txt",
            mime = "text/plain",
            use_container_width = True,)

# ==========================================
# SECTION-2: WHAT-IF SENSITIVITY LAB
# ==========================================

elif window_choice == "⚡ What-If Sensitivity Lab":
    st.markdown (
        "<div class='window-header'><h3>⚡Dynamic Parameter Sensitivity Simulator</h3></div>",
        unsafe_allow_html = True,)

    # Safeguard against uninitialized session state

    if "predicted_price" not in st.session_state:
        st.warning ("⚠️ Please calculate a baseline price in the Executive Command Center first !!")
        st.stop ()

    base_pred = st.session_state ["predicted_price"]
    base_kms = st.session_state ["kms_driven"]
    base_year = st.session_state ["manufacture_year"]
    base_pres = st.session_state ["present_price"]

    c1,c2 = st.columns (2)

    with c1:
        st.subheader ("🛠️ Interactive Sensitivity Controls")
        sim_kms = st.slider (
            "Simulate Total Odometer Mileage",
            1000,
            200000,
            int (base_kms),
            step = 5000,
        )
        sim_hold_years = st.slider (
            "Hold Vehicle for Additional Ownership Years",0,5,0,step = 1)

        current_year = 2026
        sim_year = base_year
        sim_age = (current_year + sim_hold_years) - sim_year
        sim_kms_total = sim_kms + (sim_hold_years * 10000)
        sim_kms_per_year = (
            sim_kms_total / sim_age if sim_age > 0 else sim_kms_total
        )

        sim_input = pd.DataFrame (
            [
                {
                    "year": sim_year,
                    "present_price": base_pres,
                    "kms_driven": sim_kms_total,
                    "fuel_type": st.session_state ["fuel_type"],
                    "seller_type": st.session_state ["seller_type"],
                    "transmission": st.session_state ["transmission"],
                    "owner": st.session_state ["owner"],
                    "vehicle_age": sim_age,
                    "is_luxury_brand": 0,
                    "kms_per_year": sim_kms_per_year,
                }
            ]
        )

        sim_price = float (pipeline.predict (sim_input)[0])
        delta_price = sim_price - base_pred

        st.markdown ("---")
        st.subheader ("📊 Dynamic Scenario Metrics")
        mc1,mc2,mc3 = st.columns (3)
        mc1.markdown (
            f'<div class="metric-card"><div class="metric-title">Base Price</div><div class="metric-value">₹ {base_pred:.2f} L</div></div>',
            unsafe_allow_html = True,)

        mc2.markdown (
            f'<div class="metric-card"><div class="metric-title">Simulated Price</div><div class="metric-value">₹ {sim_price:.2f} L</div></div>',
            unsafe_allow_html = True,)

        color_val = "#00D26A" if delta_price >= 0 else "#FF4B4B"
        mc3.markdown (
            f'<div class="metric-card"><div class="metric-title">Value Delta</div><div class="metric-value" style="color: {color_val};">₹ {delta_price:+.2f} L</div></div>',
            unsafe_allow_html = True,)

    with c2:
        st.subheader ("📈 Scenario Valuation Shift Chart")
        pct_shift = ((sim_price - base_pred) / base_pred * 100) if base_pred > 0 else 0.0
        st.info (
            f"💡 **Sensitivity Analysis:** Adjusting driven distance to **{sim_kms_total:,} km** and holding for **{sim_hold_years} additional year(s)** produces a **{pct_shift:+.1f} %** shift in market valuation")

        delta_df = pd.DataFrame (
            {
                "Valuation Scenario": [
                    "Base Evaluation",
                    "Simulated Ownership Scenario",
                ],
                "Valuation (Lakhs ₹)": [base_pred,sim_price],
            }
        )
        fig_delta = px.bar (
            delta_df,
            x = "Valuation Scenario",
            y = "Valuation (Lakhs ₹)",
            color = "Valuation Scenario",
            color_discrete_map = {
                "Base Evaluation": "#29B6F6",
                "Simulated Ownership Scenario": "#00D26A" if delta_price >= 0 else "#FF4B4B",
            },
            template = "plotly_dark",
            height = 340,
        )
        fig_delta.update_layout (showlegend = False,margin = dict(l = 20,r = 20,t = 30,b = 20))
        st.plotly_chart (fig_delta,use_container_width = True)

# ==========================================
# SECTION-3: SIDE-BY-SIDE COMPARISON LAB
# ==========================================

elif window_choice == "⚔️ Side By Side Comparison":
    st.markdown (
        "<div class='window-header'><h3>⚔️ Side By Side Vehicle Benchmarking</h3></div>",
        unsafe_allow_html = True,)

    c1,c2 = st.columns (2)

    with c1:
        st.subheader ("🚘 First Vehicle Configuration")
        name_a = st.text_input ("Model Name",value = "Maruti Swift")
        year_a = st.slider ("Manufacturing Year",2005,2026,2018,key = "y_a")
        pres_a = st.number_input (
            "Showroom Price (Lakhs)",0.5,100.0,7.5,key = "p_a")
        kms_a = st.number_input (
            "Kilometers Driven",100,300000,45000,key = "k_a")
        fuel_a = st.selectbox ("Fuel Type",["Petrol","Diesel","CNG"],key = "f_a")
        trans_a = st.selectbox (
            "Transmission Mode",["Manual","Automatic"],key = "t_a")

    with c2:
        st.subheader ("🚘 Second Vehicle Configuration")
        name_b = st.text_input ("Model Name",value = "Honda City")
        year_b = st.slider ("Manufacturing Year",2005,2026,2019,key = "y_b")
        pres_b = st.number_input (
            "Showroom Price (Lakhs)",0.5,100.0,12.0,key = "p_b")
        kms_b = st.number_input (
            "Kilometers Driven",100,300000,32000,key = "k_b")
        fuel_b = st.selectbox ("Fuel Type",["Petrol","Diesel","CNG"],key = "f_b")
        trans_b = st.selectbox (
            "Transmission Mode",["Manual","Automatic"],key = "t_b")

    def eval_car (year,pres,kms,fuel,trans):
        age = 2026 - year
        kpy = kms / age if age > 0 else kms
        inp = pd.DataFrame (
            [
                {
                    "year": year,
                    "present_price": pres,
                    "kms_driven": kms,
                    "fuel_type": fuel,
                    "seller_type": "Dealer",
                    "transmission": trans,
                    "owner": 0,
                    "vehicle_age": age,
                    "is_luxury_brand": 0,
                    "kms_per_year": kpy,
                }
            ]
        )
        return float (pipeline.predict (inp)[0])

    val_a = eval_car (year_a,pres_a,kms_a,fuel_a,trans_a)
    val_b = eval_car (year_b,pres_b,kms_b,fuel_b,trans_b)

    comp_df = pd.DataFrame (
        {
            "Metric": [
                "Resale Valuation (Lakhs ₹)",
                "Retained Value (%)",
                "Depreciation (Lakhs ₹)",
            ],
            f"{name_a}": [val_a,(val_a / pres_a)*100,pres_a-val_a],
            f"{name_b}": [val_b,(val_b / pres_b)*100,pres_b-val_b],
        }
    )

    st.markdown ("---")
    st.subheader ("📊 Head To Head Benchmarking Matrix Representation")

    # FIXED: Apply styling only to numeric columns to prevent type error with string column 'Metric'

    st.dataframe (
        comp_df.style.highlight_max (
            axis = 1,subset = [f"{name_a}",f"{name_b}"],color = "#238636"
        ).format ({f"{name_a}": "{:.2f}",f"{name_b}": "{:.2f}"}),
        use_container_width = True,)

    fig_comp = px.bar (
        comp_df,
        x = "Metric",
        y = [f"{name_a}",f"{name_b}"],
        barmode = "group",
        template = "plotly_dark",
        height = 350,
    )
    st.plotly_chart (fig_comp,use_container_width = True)

# ==========================================
# SECTION-4: BUY VS LEASE DECISION MATRIX
# ==========================================

elif window_choice == "📊 Buy vs Lease Decision Matrix":
    st.markdown (
        "<div class='window-header'><h3>📊 Buy Used vs Buy New vs 3-Year Lease Comparison Analysis</h3></div>",
        unsafe_allow_html = True,)

    pred_price = st.session_state ["predicted_price"]
    pres_price = st.session_state ["present_price"]

    c1,c2,c3 = st.columns (3)

    with c1:
        st.subheader ("🚗 Option A: Buy Used Model")
        used_outlay = pred_price*100000
        used_dep = used_outlay*0.25
        st.metric ("Upfront Capital Outlay",f"₹ {used_outlay/1e5:.2f} Lakhs")
        st.metric (
            "3-Year Expected Depreciation",f"₹ {used_dep/1e5:.2f} Lakhs")

    with c2:
        st.subheader ("🚘 Option B: Buy Brand New Model")
        new_outlay = pres_price*100000
        new_dep = new_outlay*0.40
        st.metric ("Upfront Capital Outlay",f"₹ {new_outlay/1e5:.2f} Lakhs")
        st.metric ("3-Year Expected Depreciation",f"₹ {new_dep/1e5:.2f} Lakhs")

    with c3:
        st.subheader ("📝 Option C: 3-Year Operating Lease")
        monthly_lease = pres_price*100000*0.02
        total_lease_cost = monthly_lease*36
        st.metric ("Monthly Lease Payment",f"₹ {monthly_lease:,.0f}")
        st.metric (
            "Total 3-Year Lease Outflow",f"₹ {total_lease_cost/1e5:.2f} Lakhs")

    st.markdown ("---")
    st.subheader ("💡 Financial Optimization Summary Representation")
    if used_dep < new_dep and used_dep < total_lease_cost:
        st.success(
            f"✅ **Buying used model is financially optimal**: You save approximately ₹ {(new_dep-used_dep)/1e5:.2f} Lakhs in capital depreciation as compared to buying a new model over 3 years")

    opt_df = pd.DataFrame (
        {
            "Financial Commitment Option": [
                "Buy Used Model",
                "Buy Brand New",
                "3-Year Operating Lease",
            ],
            "Net Cost Outflow (Lakhs ₹)": [
                used_dep / 1e5,
                new_dep / 1e5,
                total_lease_cost / 1e5,
            ],
        }
    )
    fig_opt = px.bar (
        opt_df,
        x = "Financial Commitment Option",
        y = "Net Cost Outflow (Lakhs ₹)",
        color = "Financial Commitment Option",
        template = "plotly_dark",
        height = 320,
    )
    st.plotly_chart (fig_opt,use_container_width = True)

# ==========================================
# SECTION-5: REGIONAL RTO & INFLATION
# ==========================================

elif window_choice == "🌐 Regional RTO & Inflation":
    st.markdown (
        "<div class='window-header'><h3>🌐 Regional RTO Tax & CPI Inflation Real Value Adjuster</h3></div>",
        unsafe_allow_html = True,)

    pred_price = st.session_state ["predicted_price"]
    mfg_year = st.session_state ["manufacture_year"]

    col1,col2 = st.columns (2)

    with col1:
        st.subheader ("🏛️ Regional RTO Transfer & Registration Tax Analysis")
        state_rto = st.selectbox (
            "Select Registration State Jurisdiction",
            [
                "Maharashtra",
                "Delhi",
                "Karnataka",
                "Tamil Nadu",
                "West Bengal",
                "Gujarat",
            ],
        )

        tax_rates = {
            "Maharashtra": 0.12,
            "Delhi": 0.10,
            "Karnataka": 0.15,
            "Tamil Nadu": 0.13,
            "West Bengal": 0.09,
            "Gujarat": 0.08,
        }
        rto_tax = (pred_price*100000)*tax_rates [state_rto]
        transfer_fee = 3500

        st.write (
            f"* **Estimated State Tax Rate ({state_rto}):** {tax_rates [state_rto]*100} %"
        )
        st.write (f"* **State RTO Road Tax:** ₹ {rto_tax:,.0f}")
        st.write (f"* **Ownership Transfer Registration Fee:** ₹ {transfer_fee:,.0f}")
        st.metric (
            "Total Regional Administrative Cost",
            f"₹ {(rto_tax + transfer_fee)/1e5:.2f} Lakhs",
        )

    with col2:
        st.subheader ("📈 CPI Inflation Adjusted Real Value Analysis")
        inflation_rate = 0.055
        years_diff = 2026 - mfg_year
        inflation_factor = (1 + inflation_rate) ** years_diff
        real_value = pred_price / inflation_factor

        st.write (f"* **Vehicle Manufacturing Year:** {mfg_year}")
        st.write (
            f"* **Cumulative CPI Inflation ({years_diff} yrs):** {((inflation_factor - 1)*100):.1f} %"
        )
        st.metric (
            "Inflation Adjusted Real Market Value",f"₹ {real_value:.2f} Lakhs"
        )

# ==========================================
# SECTION-6: NEGOTIATION & TCO MATRIX
# ==========================================

elif window_choice == "🤝 Negotiation & TCO Matrix":
    st.markdown (
        "<div class='window-header'><h3>🤝 Buyer/Seller Negotiation Matrix & 3-Year TCO Estimator Representation</h3></div>",
        unsafe_allow_html = True,)

    pred_price = st.session_state ["predicted_price"]

    c1,c2 = st.columns (2)

    with c1:
        st.subheader ("💰 Recommended Price Strategy Bands")
        great_deal = pred_price*0.92
        fair_deal = pred_price
        overpriced = pred_price*1.08

        st.markdown (
            f"""
        * Great Buyer Target Price: <span class='badge-deal'>₹ {great_deal:.2f} Lakhs</span>\n
        * Fair Resale Market Valuation: <span class='badge-deal'>₹ {fair_deal:.2f} Lakhs\n
        * Overpriced Threshold: <span class='badge-overprice'>₹ {overpriced:.2f} Lakhs</span>
        """,
            unsafe_allow_html = True,)

        st.markdown ("### 💡 Strategic Negotiation Playbook")
        if st.session_state ["kms_driven"] > 50000:
            st.write (
                "📉 **Buyer Leverage:** High odometer reading (> 50k km) justifies negotiating a 5–8% discount based on upcoming component wear"
            )
        else:
            st.write (
                "📈 **Seller Advantage:** Low usage (< 50k km) supports listing at the top of the price range")

    with c2:
        st.subheader ("⛽ 3-Year Total Cost Of Ownership (TCO)")
        annual_kms = st.slider (
            "Expected Annual Distance (KM)",3000,30000,10000)

        fuel = st.session_state ["fuel_type"]
        fuel_price = (
            100.0 if fuel == "Petrol" else (90.0 if fuel == "Diesel" else 75.0)
        )
        mileage = (
            15.0 if fuel == "Petrol" else (18.0 if fuel == "Diesel" else 22.0)
        )

        annual_fuel_cost = (annual_kms / mileage)*fuel_price
        annual_maint = (
            25000 if st.session_state ["present_price"] > 15 else 10000
        )

        tco_df = pd.DataFrame (
            {
                "Expense Category": [
                    "Vehicle Purchase",
                    "3-Year Fuel Expense",
                    "3-Year Maintenance & Insurance",
                ],
                "Cost (₹)": [
                    pred_price*100000,
                    3*annual_fuel_cost,
                    3*annual_maint,
                ],
            }
        )

        fig_tco = px.pie (
            tco_df,
            values = "Cost (₹)",
            names = "Expense Category",
            hole = 0.4,
            template = "plotly_dark",
            title = "3-Year Financial Outflow Distribution Analysis",
        )
        fig_tco.update_layout (margin = dict(l = 10,r = 10,t = 30,b = 10))
        st.plotly_chart (fig_tco,use_container_width = True)

# ==========================================
# SECTION-7: MARKET POSITION BENCHMARK
# ==========================================

elif window_choice == "📈 Market Position Benchmark":
    st.markdown (
        "<div class='window-header'><h3>📈 Market Positioning Scatter Matrix</h3></div>",
        unsafe_allow_html = True,)

    pred_price = st.session_state ["predicted_price"]
    pres_price = st.session_state ["present_price"]

    fig_bench = px.scatter (
        df,
        x = "present_price",
        y = "selling_price",
        color = "fuel_type",
        size = "kms_driven",
        hover_name = "car_name",
        template = "plotly_dark",
        height = 520,
        title = "Market Positioning Matrix vs Historical Dataset",
        labels = {
            "present_price": "Showroom Price (Lakhs ₹)",
            "selling_price": "Selling Price (Lakhs ₹)",
        },
    )

    fig_bench.add_trace (
        go.Scatter (
            x = [pres_price],
            y = [pred_price],
            mode = "markers",
            marker = dict(size = 22,color = "#00D26A",symbol = "star"),
            name = "Your Evaluated Vehicle",
        )
    )

    fig_bench.update_layout (margin = dict(l = 10,r = 10,t = 35,b = 10))
    st.plotly_chart (fig_bench,use_container_width = True)

# ==========================================
# SECTION-8: VISUAL ANALYTICS LAB
# ==========================================

else:
    st.markdown (
        "<div class='window-header'><h3>🔬 Visual Analytics & Correlation Laboratory</h3></div>",
        unsafe_allow_html = True,)

    # Grid Row 1

    c1,c2 = st.columns (2)
    with c1:
        st.subheader ("Showroom Price vs Resale Price Distribution Analysis")
        fig_scatter = px.scatter (
            df,
            x = "present_price",
            y = "selling_price",
            color = "fuel_type",
            size = "vehicle_age",
            hover_name = "car_name",
            template = "plotly_dark",
            height = 360,
            labels = {
                "present_price": "Showroom Price (Lakhs ₹)",
                "selling_price": "Selling Price (Lakhs ₹)",
            },
        )
        st.plotly_chart (fig_scatter,use_container_width = True)

    with c2:
        st.subheader ("Average Selling Price by Fuel & Transmission")
        avg_df = (
            df.groupby (["fuel_type","transmission"])["selling_price"]
            .mean ()
            .reset_index ()
        )
        fig_bar = px.bar (
            avg_df,
            x = "fuel_type",
            y = "selling_price",
            color = "transmission",
            barmode = "group",
            template = "plotly_dark",
            height = 360,
            labels = {
                "selling_price": "Mean Resale Price (Lakhs ₹)",
                "fuel_type": "Fuel Type",
            },
        )
        st.plotly_chart (fig_bar,use_container_width = True)

    st.markdown ("---")

    # Grid Row 2

    c3,c4 = st.columns (2)
    with c3:
        st.subheader ("Kilometers Driven vs Resale Depreciation Trend Analysis")
        fig_kms = px.scatter (
            df,
            x = "kms_driven",
            y = "selling_price",
            color = "seller_type",
            trendline = "ols",
            template = "plotly_dark",
            height = 360,
            labels = {
                "kms_driven": "Total Odometer (KM)",
                "selling_price": "Selling Price (Lakhs ₹)",
            },
        )
        st.plotly_chart (fig_kms,use_container_width = True)

    with c4:
        st.subheader ("Feature Correlation Matrix Representation")
        numeric_cols = [
            "selling_price",
            "present_price",
            "kms_driven",
            "vehicle_age",
            "owner",
        ]
        corr_matrix = df [numeric_cols].corr ().round (2)

        fig_corr = px.imshow (
            corr_matrix,
            text_auto = True,
            color_continuous_scale = "Greens",
            template = "plotly_dark",
            height = 360,
            aspect = "auto",
        )
        st.plotly_chart (fig_corr,use_container_width = True)

    st.markdown ("---")

    # Grid Row 3

    c5,c6 = st.columns (2)
    with c5:
        st.subheader ("Price Volatility by Vehicle Age Bracket")
        df ["Age_Group"] = pd.cut (
            df ["vehicle_age"],
            bins = [-1,3,7,11,30],
            labels = ["0-3 Years","4-7 Years","8-11 Years","12+ Years"],
        )
        fig_box = px.box (
            df,
            x = "Age_Group",
            y = "selling_price",
            color = "Age_Group",
            points = "all",
            template = "plotly_dark",
            height = 360,
            labels = {
                "Age_Group": "Vehicle Age Group",
                "selling_price": "Selling Price (Lakhs ₹)",
            },
        )
        fig_box.update_layout (showlegend = False)
        st.plotly_chart (fig_box,use_container_width = True)

    with c6:
        st.subheader ("Seller Type Market Share & Inventory Proportion Analysis")
        seller_counts = df ["seller_type"].value_counts ().reset_index ()
        seller_counts.columns = ["Seller Type","Count"]

        fig_donut = px.pie (
            seller_counts,
            values = "Count",
            names = "Seller Type",
            hole = 0.45,
            template = "plotly_dark",
            height = 360,
            color_discrete_sequence = px.colors.qualitative.Set2,
        )
        st.plotly_chart (fig_donut,use_container_width = True)