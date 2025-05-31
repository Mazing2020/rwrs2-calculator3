
import streamlit as st
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("final_gb_model.pkl")
encoder = joblib.load("draft_bin_encoder.pkl")

# Helper scoring functions
def draft_score(pick):
    return (1 - pick / 256) * 100

def early_declare_score(val):
    return 100 if val else 50

def breakout_age_score(age):
    if age <= 19:
        return 100
    elif age == 20:
        return 80
    elif age == 21:
        return 60
    else:
        return 40

# Streamlit UI
st.title("RWRS²: Rookie Wide Receiver Success Score")

st.write("Enter rookie WR traits to calculate RWRS² score and predict success:")

draft_pick = st.number_input("Draft Pick Number (e.g. 22)", min_value=1, max_value=256, value=22)
early_declare = st.selectbox("Declared Early?", ["Yes", "No"]) == "Yes"
breakout_age = st.slider("Breakout Age", 18, 23, 20)
dominator = st.slider("College Dominator Score (0-100)", 0, 100, 85)
athleticism = st.slider("Athleticism Score (0-100)", 0, 100, 90)
route_running = st.slider("Route Running Score (0-100)", 0, 100, 88)
landing_spot = st.slider("NFL Landing Spot Fit (0-100)", 0, 100, 80)

# Compute features
features = [
    draft_score(draft_pick),
    early_declare_score(early_declare),
    breakout_age_score(breakout_age),
    dominator,
    athleticism,
    route_running,
    landing_spot
]

interaction = features[2] * features[4]
draft_bin = encoder.transform(np.array([[features[0]]]))
X_input = np.concatenate([features, [interaction], draft_bin.flatten()]).reshape(1, -1)

# Predict
if st.button("Calculate RWRS² Score"):
    score = model.predict(X_input)[0]
    st.success(f"Predicted RWRS² Score: {round(score, 2)} (Lower is Better)")
    st.caption("RWRS² is based on similarity to top 10 WRs ranked by user-defined metrics.")
