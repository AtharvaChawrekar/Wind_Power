import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wind Turbine Power Predictor", layout="wide")
st.title("🌬️ Wind Turbine Power Output Predictor")

@st.cache_resource
def load_artifacts():
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
    features = meta.get('features', [])
    target = meta.get('target', 'ActivePower')
    seq_len = meta.get('seq_length', 24)
    st.sidebar.success("✅ Models & Artifacts Loaded")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Sidebar options
st.sidebar.header("Settings")
selected_model_name = st.sidebar.selectbox("Select Model Architecture", ["LSTM", "CNN", "MLP"])
input_mode = st.sidebar.radio("Input Method", ["🎛️ Manual Slider Entry", "📁 Batch CSV Upload"])

def run_inference(model_name, X_seq):
    if model_name == "LSTM":
        preds = lstm_model.predict(X_seq, verbose=0)
    elif model_name == "CNN":
        preds = cnn_model.predict(X_seq, verbose=0)
    else:
        preds = mlp_model.predict(X_seq.reshape((X_seq.shape[0], -1)), verbose=0)
    return preds

# ==========================================
# OPTION 1: MANUAL SLIDER ENTRY
# ==========================================
if input_mode == "🎛️ Manual Slider Entry":
    st.subheader("Interactive Manual Parameter Simulation")
    st.markdown("Simulate environmental and mechanical conditions to predict real-time **ActivePower (kW)**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 💨 Aerodynamics & Wind")
        wind_speed = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=9.5, step=0.1)
        wind_dir = st.slider("Wind Direction (°)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
        pitch_angle = st.slider("Blade Pitch Angle (°)", min_value=-5.0, max_value=90.0, value=1.2, step=0.1)
        nacelle_pos = st.slider("Nacelle Position (°)", min_value=0.0, max_value=360.0, value=182.0, step=1.0)

    with col2:
        st.markdown("##### ⚙️ Rotational Subsystems")
        rotor_rpm = st.slider("Rotor RPM", min_value=0.0, max_value=25.0, value=15.2, step=0.1)
        gen_rpm = st.slider("Generator RPM", min_value=0.0, max_value=2000.0, value=1520.0, step=10.0)
        ambient_temp = st.slider("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
        reactive_pwr = st.slider("Reactive Power (kVAR)", min_value=-200.0, max_value=400.0, value=15.0, step=5.0)

    with col3:
        st.markdown("##### 🌡️ Core Temperatures")
        gearbox_temp = st.slider("Gearbox Bearing Temp (°C)", min_value=20.0, max_value=100.0, value=65.0, step=0.5)
        gen_winding_temp = st.slider("Generator Winding Temp (°C)", min_value=20.0, max_value=120.0, value=70.0, step=0.5)
        bearing_shaft_temp = st.slider("Bearing Shaft Temp (°C)", min_value=20.0, max_value=90.0, value=48.0, step=0.5)
        turbine_status = st.selectbox("Turbine Status", [1, 0], format_func=lambda x: "Operational (1)" if x == 1 else "Offline / Fault (0)")

    if st.button("🚀 Predict Power Output", type="primary"):
        # Map user input values to the 19 expected features
        manual_dict = {
            'WindSpeed': wind_speed,
            'WindDirection': wind_dir,
            'Blade1PitchAngle': pitch_angle,
            'Blade2PitchAngle': pitch_angle,
            'Blade3PitchAngle': pitch_angle,
            'RotorRPM': rotor_rpm,
            'GeneratorRPM': gen_rpm,
            'AmbientTemperatue': ambient_temp,
            'BearingShaftTemperature': bearing_shaft_temp,
            'GearboxBearingTemperature': gearbox_temp,
            'GearboxOilTemperature': gearbox_temp - 5.0,
            'GeneratorWinding1Temperature': gen_winding_temp,
            'GeneratorWinding2Temperature': gen_winding_temp - 2.0,
            'HubTemperature': ambient_temp + 5.0,
            'ControlBoxTemperature': ambient_temp + 8.0,
            'MainBoxTemperature': ambient_temp + 6.0,
            'NacellePosition': nacelle_pos,
            'ReactivePower': reactive_pwr,
            'TurbineStatus': turbine_status
        }

        # Build single row and add target dummy column for scaler transform
        row_df = pd.DataFrame([{col: manual_dict.get(col, 0.0) for col in features}])
        row_df[target] = 0.0

        # Scale features
        scaled_row = scaler.transform(row_df)[0]
        feature_vals = scaled_row[:-1]  # Exclude target column

        # Replicate row across seq_len time steps to satisfy sequence lookback
        X_manual = np.tile(feature_vals, (1, seq_len, 1))

        # Run selected model
        raw_pred = run_inference(selected_model_name, X_manual)

        # Inverse transform prediction to kW
        pred_full = np.zeros((1, len(features) + 1))
        pred_full[0, -1] = raw_pred[0][0]
        unscaled_power = scaler.inverse_transform(pred_full)[0, -1]
        unscaled_power = max(0.0, unscaled_power) if turbine_status == 1 else 0.0

        st.divider()
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Predicted Active Power", f"{unscaled_power:,.2f} kW")
        m_col2.metric("Architecture Used", selected_model_name)
        m_col3.metric("Capacity Factor (of 1.8MW)", f"{(unscaled_power / 1800.0) * 100:.1f}%")

        # Visual gauge / operating point
        fig, ax = plt.subplots(figsize=(8, 3))
        curve_winds = np.linspace(0, 25, 100)
        # Standard sigmoidal wind power curve approximation for reference
        curve_pwr = np.where(curve_winds < 3.0, 0, 1800 / (1 + np.exp(-0.7 * (curve_winds - 9))))
        curve_pwr = np.where(curve_winds > 20.0, 0, curve_pwr)

        ax.plot(curve_winds, curve_pwr, color="gray", linestyle="--", label="Theoretical Power Curve")
        ax.scatter([wind_speed], [unscaled_power], color="crimson", s=120, zorder=5, label="Current Operating Point")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Turbine Operating Point on Power Curve")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

# ==========================================
# OPTION 2: BATCH CSV UPLOAD
# ==========================================
else:
    st.subheader("Batch CSV Telemetry Forecast")
    uploaded_file = st.file_uploader("Upload SCADA CSV File", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("##### Data Preview (First 5 Rows)", df_input.head())

        missing = [col for col in features if col not in df_input.columns]
        if missing:
            st.error(f"Missing required feature columns in CSV: {missing}")
            st.info("💡 You can switch to **'🎛️ Manual Slider Entry'** in the sidebar to test predictions without this CSV.")
        else:
            df_clean = df_input[features].dropna()
            dummy_df = df_clean.copy()
            dummy_df[target] = 0
            scaled = scaler.transform(dummy_df)

            if len(scaled) < seq_len:
                st.warning(f"Uploaded CSV needs at least {seq_len} rows for sequence generation.")
            else:
                X_infer = []
                for i in range(len(scaled) - seq_len + 1):
                    X_infer.append(scaled[i:i + seq_len, :-1])
                X_infer = np.array(X_infer)

                preds = run_inference(selected_model_name, X_infer)

                pred_full = np.zeros((len(preds), len(features) + 1))
                pred_full[:, -1] = preds.flatten()
                unscaled_preds = scaler.inverse_transform(pred_full)[:, -1]

                st.write(f"##### Predicted ActivePower Output ({selected_model_name})")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(unscaled_preds, label=f"Predicted Power ({selected_model_name})", color="#0072B2")
                ax.set_ylabel("Power Output (kW)")
                ax.set_xlabel("Time Step Index")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                st.pyplot(fig)

                results_df = pd.DataFrame({"Predicted ActivePower (kW)": unscaled_preds})
                st.dataframe(results_df)
    else:
        st.info("Upload a CSV file containing turbine telemetry, or select **Manual Slider Entry** in the sidebar.")
