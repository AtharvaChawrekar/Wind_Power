import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wind Power Prediction Dashboard", layout="wide")
st.title("🌬️ Wind Turbine Power Output Predictor")

# 1. Load trained models and metadata
@st.cache_resource
def load_artifacts():
    # Adding compile=False avoids deserialization errors for losses/metrics during inference
    mlp = tf.keras.models.load_model('model_mlp.h5', compile=False)
    cnn = tf.keras.models.load_model('model_cnn.h5', compile=False)
    lstm = tf.keras.models.load_model('model_lstm.h5', compile=False)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
        
    return mlp, cnn, lstm, scaler, meta

try:
    mlp_model, cnn_model, lstm_model, scaler, meta = load_artifacts()
    
    # Safe retrieval of keys from meta.pkl
    features = meta.get('features', [])
    target = meta.get('target', 'ActivePower')
    seq_len = meta.get('seq_length', 24)
    
    st.sidebar.success("Models and Scalers Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 2. Sidebar controls
selected_model_name = st.sidebar.selectbox("Choose Model Architecture", ["LSTM", "CNN", "MLP"])
uploaded_file = st.sidebar.file_uploader("Upload CSV Data for Inference", type=["csv"])

# 3. Prediction & Visualization Logic
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df_input.head())

    # Check for missing feature columns
    missing = [col for col in features if col not in df_input.columns]
    if missing:
        st.error(f"Missing required feature columns in CSV: {missing}")
    else:
        df_clean = df_input[features].dropna()
        
        # Build dummy frame to inverse scale the predictions later
        dummy_df = df_clean.copy()
        dummy_df[target] = 0
        scaled = scaler.transform(dummy_df)
        
        if len(scaled) < seq_len:
            st.warning(f"Uploaded CSV needs at least {seq_len} rows for sequence generation.")
        else:
            # Build sequence windows
            X_infer = []
            for i in range(len(scaled) - seq_len + 1):
                X_infer.append(scaled[i:i + seq_len, :-1])
            X_infer = np.array(X_infer)

            # Generate predictions
            if selected_model_name == "LSTM":
                preds = lstm_model.predict(X_infer)
            elif selected_model_name == "CNN":
                preds = cnn_model.predict(X_infer)
            else:
                preds = mlp_model.predict(X_infer.reshape((X_infer.shape[0], -1)))

            # Inverse scale predictions to real kW values
            pred_full = np.zeros((len(preds), len(features) + 1))
            pred_full[:, -1] = preds.flatten()
            unscaled_preds = scaler.inverse_transform(pred_full)[:, -1]

            # Plot results
            st.write(f"### Predicted {target} Output ({selected_model_name})")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(unscaled_preds, label=f"Predicted {target}", color="#0072B2")
            ax.set_ylabel("Power Output (kW)")
            ax.set_xlabel("Time Step Index")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

            # Display table
            results_df = pd.DataFrame({f"Predicted {target} (kW)": unscaled_preds})
            st.dataframe(results_df)
else:
    st.info("Upload a CSV file containing turbine telemetry features to generate power predictions.")