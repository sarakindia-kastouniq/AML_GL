"""
app.py – Streamlit Frontend
============================
Loads the best model from Hugging Face model hub and serves a
prediction UI for the Wellness Tourism Package classifier.

Run locally:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# CONFIG  ← update HF_USERNAME before deploying
# ─────────────────────────────────────────────
HF_USERNAME  = os.environ.get("HF_USERNAME", "your_hf_username_here")
MODEL_REPO   = f"{HF_USERNAME}/tourism-wellness-model"
MODEL_FILE   = "best_model.pkl"

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Wellness Tourism Package Predictor")
st.markdown(
    "Predict whether a customer is likely to purchase the **Wellness Tourism Package** "
    "before a sales call is made."
)
st.divider()


# ─────────────────────────────────────────────
# LOAD MODEL (cached so it is only downloaded once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face …")
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model()
st.success("Model loaded successfully.", icon="✅")

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age                      = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact          = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    city_tier                = st.selectbox("City Tier", [1, 2, 3])
    occupation               = st.selectbox("Occupation",
                                            ["Salaried", "Small Business", "Large Business",
                                             "Free Lancer"])
    gender                   = st.selectbox("Gender", ["Male", "Female"])
    num_persons_visiting     = st.number_input("Number of Persons Visiting", 1, 10, 2)
    preferred_property_star  = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status           = st.selectbox("Marital Status",
                                            ["Single", "Married", "Divorced", "Unmarried"])

with col2:
    num_trips                = st.number_input("Number of Trips (avg/year)", 0, 20, 3)
    passport                 = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car                  = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children_visiting    = st.number_input("Number of Children (< 5 yrs)", 0, 5, 0)
    designation              = st.selectbox("Designation",
                                            ["Executive", "Manager", "Senior Manager",
                                             "AVP", "VP"])
    monthly_income           = st.number_input("Monthly Income (₹)", 5000, 200000, 50000, step=1000)
    pitch_satisfaction       = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched          = st.selectbox("Product Pitched",
                                            ["Basic", "Standard", "Deluxe",
                                             "Super Deluxe", "King"])
    num_followups            = st.number_input("Number of Follow-ups", 0, 10, 3)
    duration_of_pitch        = st.number_input("Duration of Pitch (minutes)", 5, 60, 15)

st.divider()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# (must mirror data_preparation.py exactly)
# ─────────────────────────────────────────────

def build_input_df() -> pd.DataFrame:
    """Convert UI values into the feature vector the model expects."""

    # Binary / ordinal encodings (same as data_preparation.py)
    gender_enc          = 1 if gender == "Male" else 0
    type_contact_enc    = 1 if type_of_contact == "Company Invited" else 0
    passport_enc        = 1 if passport == "Yes" else 0
    own_car_enc         = 1 if own_car == "Yes" else 0

    # One-hot encode Occupation
    occ_options = ["Small Business", "Large Business", "Free Lancer"]  # 'Salaried' is baseline
    occ_encoded = {f"Occupation_{o.replace(' ', '_')}": int(occupation == o)
                   for o in occ_options}

    # One-hot encode MaritalStatus  (baseline: 'Divorced')
    ms_options = ["Married", "Single", "Unmarried"]
    ms_encoded = {f"MaritalStatus_{m}": int(marital_status == m) for m in ms_options}

    # One-hot encode Designation  (baseline: 'AVP')
    des_options = ["Executive", "Manager", "Senior Manager", "VP"]
    des_encoded = {f"Designation_{d.replace(' ', '_')}": int(designation == d)
                   for d in des_options}

    # One-hot encode ProductPitched  (baseline: 'Basic')
    pp_options = ["Deluxe", "King", "Standard", "Super Deluxe"]
    pp_encoded = {f"ProductPitched_{p.replace(' ', '_')}": int(product_pitched == p)
                  for p in pp_options}

    base = {
        "Age":                     age,
        "TypeofContact":           type_contact_enc,
        "CityTier":                city_tier,
        "DurationOfPitch":         duration_of_pitch,
        "Gender":                  gender_enc,
        "NumberOfPersonVisiting":  num_persons_visiting,
        "NumberOfFollowups":       num_followups,
        "PreferredPropertyStar":   preferred_property_star,
        "NumberOfTrips":           num_trips,
        "Passport":                passport_enc,
        "PitchSatisfactionScore":  pitch_satisfaction,
        "OwnCar":                  own_car_enc,
        "NumberOfChildrenVisiting": num_children_visiting,
        "MonthlyIncome":           monthly_income,
    }
    base.update(occ_encoded)
    base.update(ms_encoded)
    base.update(des_encoded)
    base.update(pp_encoded)

    return pd.DataFrame([base])


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
if st.button("🔍 Predict Purchase Likelihood", use_container_width=True, type="primary"):
    input_df = build_input_df()

    # Align columns with what the model was trained on (fills any missing with 0)
    try:
        model_features = model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)
    except AttributeError:
        # Some estimators don't expose feature_names_in_; proceed as-is
        pass

    prediction   = model.predict(input_df)[0]
    probability  = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(
            f"**Likely to Purchase** ✅\n\n"
            f"Predicted probability: **{probability:.1%}**",
            icon="🎯",
        )
    else:
        st.warning(
            f"**Unlikely to Purchase** ❌\n\n"
            f"Predicted probability: **{probability:.1%}**",
            icon="📉",
        )

    st.caption(
        "Confidence is derived from the model's predicted probability. "
        "A score above 50% indicates a positive purchase signal."
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Built for *Visit with Us* · MLOps Project · "
    "Powered by Scikit-learn & Streamlit</small>",
    unsafe_allow_html=True,
)
