import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Cloud Cost Optimizer", layout="centered")

st.title("☁️ Cloud Cost Optimization Dashboard")

st.write("""
This dashboard predicts cloud cost based on usage
and provides optimization recommendations.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload Cloud Usage CSV", type=["csv"])

# Run only if file is uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.head())

        # Required columns check
        required_cols = ["cpu_hrs", "memory_gb", "storage_gb", "network_gb"]
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain columns: cpu_hrs, memory_gb, storage_gb, network_gb")
        else:
            X = df[required_cols]

            # Train model (demo)
            model = RandomForestRegressor(random_state=42)
            model.fit(X, np.random.uniform(50, 500, len(X)))

            # Predictions
            df["Predicted Cost"] = model.predict(X)

            st.subheader("💰 Predicted Cloud Cost")
            st.dataframe(df)

            # Feature importance
            st.subheader("🔍 Cost Driver Analysis")
            importance = model.feature_importances_
            features = ["CPU Hours", "Memory", "Storage", "Network"]

            fig, ax = plt.subplots()
            ax.barh(features, importance)
            ax.set_xlabel("Importance")
            st.pyplot(fig)

            # Recommendation
            top_feature = features[np.argmax(importance)]

            st.subheader("🛠 Optimization Recommendation")
            if top_feature == "CPU Hours":
                st.success("Reduce idle CPU usage using auto-scaling.")
            elif top_feature == "Memory":
                st.success("Avoid over-provisioned memory.")
            elif top_feature == "Storage":
                st.success("Move unused data to cold storage.")
            elif top_feature == "Network":
                st.success("Reduce cross-region data transfer.")

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded CSV file is empty.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("⬆️ Please upload a CSV file to continue.")
