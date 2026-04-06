import streamlit as st
import pickle
import numpy as np
import os

# -------------------- LOAD MODEL --------------------

BASE_DIR = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(BASE_DIR, "personality_model.pkl"), "rb"))

# -------------------- UI --------------------

st.set_page_config(page_title="Personality Predictor", layout="wide")

st.title("🧠 Personality Prediction App")
st.markdown("### Predict personality using Machine Learning")

st.sidebar.header("Input Features")

# -------------------- INPUTS (ALL 26 FEATURES) --------------------

def slider(name):
    return st.sidebar.slider(name, 0, 10, 5)

social_energy = slider("Social Energy")
alone_time_preference = slider("Alone Time Preference")
talkativeness = slider("Talkativeness")
deep_reflection = slider("Deep Reflection")
group_comfort = slider("Group Comfort")
party_liking = slider("Party Liking")
listening_skill = slider("Listening Skill")
empathy = slider("Empathy")
organization = slider("Organization")
leadership = slider("Leadership")
risk_taking = slider("Risk Taking")
public_speaking_comfort = slider("Public Speaking Comfort")
curiosity = slider("Curiosity")
routine_preference = slider("Routine Preference")
excitement_seeking = slider("Excitement Seeking")
friendliness = slider("Friendliness")
planning = slider("Planning")
spontaneity = slider("Spontaneity")
adventurousness = slider("Adventurousness")
reading_habit = slider("Reading Habit")
sports_interest = slider("Sports Interest")
online_social_usage = slider("Online Social Usage")
travel_desire = slider("Travel Desire")
gadget_usage = slider("Gadget Usage")
work_style_collaborative = slider("Work Style Collaborative")
decision_speed = slider("Decision Speed")

# -------------------- CREATE INPUT ARRAY --------------------

input_data = np.array([[
    social_energy, alone_time_preference, talkativeness, deep_reflection,
    group_comfort, party_liking, listening_skill, empathy, organization,
    leadership, risk_taking, public_speaking_comfort, curiosity,
    routine_preference, excitement_seeking, friendliness, planning,
    spontaneity, adventurousness, reading_habit, sports_interest,
    online_social_usage, travel_desire, gadget_usage,
    work_style_collaborative, decision_speed
]])

# -------------------- PREDICTION --------------------

if st.button("Predict Personality"):

    try:
        prediction = model.predict(input_data)[0]

        # Adjust labels if needed
        if prediction == 0:
            result = "Introvert 🧘"
        else:
            result = "Extrovert 🎉"

        st.success(f"Predicted Personality: **{result}**")

        # Confidence (if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            confidence = np.max(proba) * 100
            st.info(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------- FOOTER --------------------

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")