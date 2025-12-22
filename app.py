import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Cloud Cost Optimizer",
    page_icon="☁️",
    layout="wide"
)

# --------------------------------------------------
# BABY BLUE THEME
# --------------------------------------------------
st.markdown("""
<style>
.main { background-color: #e8f3ff; }
h1, h2, h3 { color: #0f4c81; }
.stButton>button {
    background-color: #4da3ff;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOGIN SYSTEM
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("☁️ Cloud Cost Optimization Dashboard")
st.caption("ML-based prediction + Real cloud pricing + Optimization insights")

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

provider = st.sidebar.selectbox(
    "Cloud Provider",
    ["AWS", "Azure"]
)

trees = st.sidebar.slider("Random Forest Trees", 50, 300, 100, 50)

# --------------------------------------------------
# PRICING MODELS (SIMPLIFIED)
# --------------------------------------------------
pricing = {
    "AWS": {
        "cpu": 4.0,
        "memory": 2.0,
        "storage": 1.5,
        "network": 1.2
    },
    "Azure": {
        "cpu": 4.5,
        "memory": 2.2,
        "storage": 1.6,
        "network": 1.3
    }
}

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Cloud Usage CSV", type=["csv"])
required_cols = ["cpu_hrs", "memory_gb", "storage_gb", "network_gb"]

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain cpu_hrs, memory_gb, storage_gb, network_gb")
        st.stop()

    X = df[required_cols]

    # --------------------------------------------------
    # REAL COST CALCULATION (PROVIDER BASED)
    # --------------------------------------------------
    p = pricing[provider]

    df["Actual Cost (₹)"] = (
        X["cpu_hrs"] * p["cpu"] +
        X["memory_gb"] * p["memory"] +
        X["storage_gb"] * p["storage"] +
        X["network_gb"] * p["network"]
    )

    # --------------------------------------------------
    # ML MODEL (LEARNS FROM REAL COST)
    # --------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=trees,
        random_state=42
    )
    model.fit(X, df["Actual Cost (₹)"])
    df["Predicted Cost (₹)"] = model.predict(X)

    # --------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------
    st.subheader("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", len(df))
    c2.metric("Avg CPU", f"{df['cpu_hrs'].mean():.1f}")
    c3.metric("Avg Memory", f"{df['memory_gb'].mean():.1f}")
    c4.metric("Avg Storage", f"{df['storage_gb'].mean():.1f}")

    # --------------------------------------------------
    # COST SUMMARY
    # --------------------------------------------------
    st.subheader("💰 Cost Summary")
    c1, c2, c3 = st.columns(3)

    c1.metric("Avg Cost", f"₹{df['Predicted Cost (₹)'].mean():.2f}")
    c2.metric("Max Cost", f"₹{df['Predicted Cost (₹)'].max():.2f}")
    c3.metric("Min Cost", f"₹{df['Predicted Cost (₹)'].min():.2f}")

    # --------------------------------------------------
    # COST TREND
    # --------------------------------------------------
    st.subheader("📈 Cost Trend")
    fig, ax = plt.subplots()
    ax.plot(df["Predicted Cost (₹)"], marker="o")
    ax.set_ylabel("Cost (₹)")
    ax.set_xlabel("Instance")
    st.pyplot(fig)

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------
    st.subheader("🔍 Cost Driver Analysis")
    importance = model.feature_importances_
    features = ["CPU", "Memory", "Storage", "Network"]

    fig2, ax2 = plt.subplots()
    ax2.barh(features, importance)
    st.pyplot(fig2)

    # --------------------------------------------------
    # COST CONTRIBUTION PIE
    # --------------------------------------------------
    st.subheader("📌 Cost Contribution (%)")
    pct = (importance / importance.sum()) * 100

    fig3, ax3 = plt.subplots()
    ax3.pie(pct, labels=features, autopct="%1.1f%%")
    ax3.axis("equal")
    st.pyplot(fig3)

    # --------------------------------------------------
    # SCENARIO SIMULATOR
    # --------------------------------------------------
    st.subheader("🧪 Optimization Simulator")
    reduction = st.slider("Reduce CPU Usage (%)", 0, 50, 10)

    sim_X = X.copy()
    sim_X["cpu_hrs"] *= (1 - reduction / 100)

    sim_cost = model.predict(sim_X).mean()
    base_cost = df["Predicted Cost (₹)"].mean()

    st.metric("Estimated Savings", f"₹{base_cost - sim_cost:.2f}")

    # --------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------
    top_feature = features[np.argmax(importance)]
    recommendations = {
        "CPU": "Enable auto-scaling or use smaller instances.",
        "Memory": "Avoid over-provisioned RAM.",
        "Storage": "Move data to cold/archive storage.",
        "Network": "Reduce cross-region traffic."
    }

    st.subheader("🛠 Optimization Recommendation")
    st.success(recommendations[top_feature])

    # --------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------
    with st.expander("🧠 How this works"):
        st.write(f"""
        • Uses **{provider} pricing model**  
        • Calculates real cost per workload  
        • ML model learns cost patterns  
        • Predicts future cost  
        • Simulates optimization impact  
        """)

    # --------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------
    st.subheader("⬇️ Downloads")

    st.download_button(
        "Download Cost Data CSV",
        df.to_csv(index=False),
        "cloud_cost_analysis.csv",
        "text/csv"
    )

    report = f"""
Cloud Cost Optimization Report

Provider: {provider}
Average Cost: ₹{base_cost:.2f}
Top Cost Driver: {top_feature}
Recommendation: {recommendations[top_feature]}
Estimated Savings: ₹{base_cost - sim_cost:.2f}
"""

    st.download_button(
        "Download Optimization Report",
        report,
        "cloud_optimization_report.txt"
    )

else:
    st.info("⬆️ Upload a CSV file to begin.")
