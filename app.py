
import pickle
import pandas as pd
import streamlit as st
from scipy.stats import boxcox


# --------------------------------------------------
# 1. LOAD TRAINED MODEL AND PREPROCESSING INFORMATION
# --------------------------------------------------

with open("heart_model.sav", "rb") as file:
    bundle = pickle.load(file)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
lambdas = bundle["lambdas"]
category_levels = bundle["category_levels"]


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cardiac Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Cardiac Disease Prediction")

st.write(
    "Enter the patient's details below and click Predict "
    "to obtain the model's prediction."
)

st.info(
    "Educational project only. This result is not a medical diagnosis."
)


# --------------------------------------------------
# 3. CATEGORY OPTIONS
# Verify these mappings against your dataset documentation.
# --------------------------------------------------

sex_options = {
    "Female": 0,
    "Male": 1
}

cp_options = {
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 3
}

fbs_options = {
    "No (120 mg/dl or below)": 0,
    "Yes (above 120 mg/dl)": 1
}

restecg_options = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}

exang_options = {
    "No": 0,
    "Yes": 1
}

slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_options = {
    0: "Thal category 0",
    1: "Normal",
    2: "Fixed defect",
    3: "Reversible defect"
}


# --------------------------------------------------
# 4. PATIENT INPUT FORM
# --------------------------------------------------

with st.form("patient_form"):

    st.subheader("1. Patient Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50
        )

        sex_label = st.selectbox(
            "Sex",
            list(sex_options.keys())
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=1,
            value=120
        )

        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=1,
            value=200
        )

        fbs_label = st.selectbox(
            "Fasting blood sugar",
            list(fbs_options.keys())
        )

        restecg_label = st.selectbox(
            "Resting ECG result",
            list(restecg_options.keys())
        )

    with col2:

        cp_label = st.selectbox(
            "Chest pain type",
            list(cp_options.keys())
        )

        thalach = st.number_input(
            "Maximum heart rate (bpm)",
            min_value=1,
            value=150
        )

        exang_label = st.selectbox(
            "Exercise-induced angina",
            list(exang_options.keys())
        )

        oldpeak = st.number_input(
            "Oldpeak (ST depression)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope_label = st.selectbox(
            "Slope of peak exercise ST segment",
            list(slope_options.keys())
        )

        ca = st.selectbox(
            "Number of major vessels",
            [0, 1, 2, 3, 4]
        )

        # Show only thal codes available in the training dataset.
        available_thal = [
            int(value)
            for value in category_levels["thal"]
            if int(value) in thal_options
        ]

        if not available_thal:
            st.error("No supported thal categories found in the model.")
            st.stop()

        thal_label = st.selectbox(
            "Thal result",
            [thal_options[value] for value in available_thal]
        )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )


# --------------------------------------------------
# 5. PREPROCESS INPUT AND MAKE PREDICTION
# --------------------------------------------------

if submitted:

    # Convert the readable selections into dataset codes.
    selected_thal = next(
        code for code in available_thal
        if thal_options[code] == thal_label
    )

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex_options[sex_label],
        "cp": cp_options[cp_label],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_options[fbs_label],
        "restecg": restecg_options[restecg_label],
        "thalach": thalach,
        "exang": exang_options[exang_label],
        "oldpeak": oldpeak,
        "slope": slope_options[slope_label],
        "ca": ca,
        "thal": selected_thal
    }])

    try:

        # Apply the same one-hot encoding used during training.
        for col in ["cp", "restecg", "thal"]:
            patient[col] = pd.Categorical(
                patient[col],
                categories=category_levels[col]
            )

        patient = pd.get_dummies(
            patient,
            columns=["cp", "restecg", "thal"],
            drop_first=True,
            dtype="uint8"
        )

        # Ensure the features and their order match training.
        patient = patient.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Apply the saved Box-Cox transformations.
        for feature, lam in lambdas.items():

            if feature not in patient.columns:
                raise ValueError(
                    f"Missing feature required for transformation: {feature}"
                )

            # Box-Cox requires strictly positive input.
            if (patient[feature] <= 0).any():
                raise ValueError(
                    f"{feature} must be positive for the saved "
                    "Box-Cox transformation. Check the exact "
                    "preprocessing used during model training."
                )

            patient[feature] = boxcox(
                patient[feature].astype(float),
                lmbda=lam
            )

        # Predict the target class.
        prediction = int(model.predict(patient)[0])

        st.subheader("2. Prediction Result")

        if prediction == 1:
            st.error("The model predicts target class 1.  that means You are Heart Patient")
        else:
            st.success("The model predicts target class 0. that means your heart condition is normal")

        st.caption(
            "Check your dataset's target definition to confirm "
            "what each class means. This prediction is for "
            "educational purposes and must not guide medical decisions."
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        st.warning(
            "Check that the saved model, feature columns, category "
            "codes, and preprocessing match your training notebook."
        )
