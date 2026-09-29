from pathlib import Path
import time
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "optimized_random_forest.pkl"

@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}\n\nRun build_model_artifact.py first.")
    return joblib.load(MODEL_PATH)

try:
    artifact = load_artifact()
    model = artifact["model"]
    FEATURES = artifact["features"]
    RANGES = artifact["feature_ranges"]
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.title("🌱 Crop Recommendation System")
st.subheader(
    "Research and Design of a Machine Learning-Based Crop Recommendation "
    "Method Integrating Environmental Factors"
)
st.write("Enter the seven environmental and soil-related parameters. The trained Random Forest returns the three highest-probability crop recommendations.")
st.info("The prototype loads the finalized optimized Random Forest artifact. It does not retrain the model when the application starts.")

st.markdown("## 1. Input Environmental Conditions")
c1, c2, c3 = st.columns(3)
with c1:
    N = st.number_input("Nitrogen (N)", 0.0, 200.0, float(RANGES["N"]["mean"]), 1.0)
    P = st.number_input("Phosphorus (P)", 0.0, 200.0, float(RANGES["P"]["mean"]), 1.0)
    K = st.number_input("Potassium (K)", 0.0, 250.0, float(RANGES["K"]["mean"]), 1.0)
with c2:
    temperature = st.number_input("Temperature (°C)", -10.0, 60.0, float(RANGES["temperature"]["mean"]), 0.1, format="%.1f")
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, float(RANGES["humidity"]["mean"]), 0.1, format="%.1f")
    ph = st.number_input("Soil pH", 0.0, 14.0, float(RANGES["ph"]["mean"]), 0.1, format="%.1f")
with c3:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, float(RANGES["rainfall"]["mean"]), 1.0)
    st.markdown("### Training-data ranges")
    for f in FEATURES:
        st.write(f"**{f}:** {RANGES[f]['min']:.2f} – {RANGES[f]['max']:.2f}")

st.markdown("## 2. Generate Recommendation")

if st.button("🌱 Recommend Crops", type="primary", use_container_width=True):
    X = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=FEATURES
    )

    start = time.perf_counter()
    probs = model.predict_proba(X)[0]
    idxs = probs.argsort()[::-1][:3]
    elapsed_ms = (time.perf_counter() - start) * 1000

    recs = [
        {
            "Rank": r,
            "Crop": str(model.classes_[i]).title(),
            "Probability": float(probs[i])
        }
        for r, i in enumerate(idxs, 1)
    ]

    st.markdown("## 3. Top-3 Crop Recommendations")

    cols = st.columns(3)

    for col, rec in zip(cols, recs):
        with col:
            st.metric(
                f"Rank {rec['Rank']}",
                rec["Crop"],
                f"{rec['Probability'] * 100:.2f}% probability"
            )

    st.table(
        pd.DataFrame(
            [
                {
                    "Rank": r["Rank"],
                    "Recommended Crop": r["Crop"],
                    "Prediction Probability": f"{r['Probability'] * 100:.2f}%"
                }
                for r in recs
            ]
        )
    )

    st.caption(f"Model inference time: {elapsed_ms:.2f} ms")

    with st.expander("View submitted input values"):
        st.dataframe(X, use_container_width=True)

    outside = [
        f for f in FEATURES
        if float(X.iloc[0][f]) < RANGES[f]["min"]
        or float(X.iloc[0][f]) > RANGES[f]["max"]
    ]

    if outside:
        st.warning(
            "Inputs outside the training-data range: "
            + ", ".join(outside)
            + ". The model will still predict, but this represents "
            "extrapolation beyond the training-data distribution."
        )

st.markdown("---")
st.markdown("## Model Information")
a, b = st.columns(2)
with a:
    st.write("**Model:** Optimized Random Forest")
    st.write("**Number of trees:** 100")
    st.write("**Maximum depth:** 10")
    st.write("**Maximum features:** sqrt")
    st.write("**Minimum samples split:** 5")
with b:
    st.write("**Input variables:** 7")
    st.write("**Crop classes:** 22")
    st.write("**Training samples:** 2,200")
    st.write("**Output:** Top-3 crops + prediction probability")

st.markdown("---")
st.warning(
    "This prototype is a research demonstration based on the available dataset. "
    "Recommendations should not be interpreted as guaranteed crop performance "
    "under field conditions. This study does not constitute agronomic advice."
)