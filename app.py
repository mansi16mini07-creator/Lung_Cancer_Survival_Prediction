import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Lung Cancer Analytics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #10141C;
        color: #F1F5F9;
    }

    [data-testid="stSidebar"] {
        background-color: #171E2B;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #A8B3C7;
        font-size: 15px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: #1B2535;
        border: 1px solid #303D52;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        background-color: #171E2B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #303D52;
    }

    .section-title {
        color: #7DD3FC;
        font-size: 22px;
        font-weight: 700;
    }

    .disclaimer {
        background-color: #392B18;
        border: 1px solid #80602C;
        padding: 14px;
        border-radius: 10px;
        color: #FFE6AE;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. LOAD TRAINED MODEL AND METADATA
# =========================================================

MODEL_PATH = Path("lung_cancer_model.pkl")
METADATA_PATH = Path("model_metadata.pkl")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model files not found. Place lung_cancer_model.pkl "
        "and model_metadata.pkl in the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

feature_columns = metadata["feature_columns"]
numeric_columns = metadata["numeric_columns"]
categorical_columns = metadata["categorical_columns"]

category_values = metadata.get("category_values", {})
numeric_defaults = metadata.get("numeric_defaults", {})

# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🫁 Lung Cancer AI")

st.sidebar.caption("Machine Learning Analytics Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patient Prediction",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "This application demonstrates a machine learning "
    "workflow for educational and research purposes."
)

# =========================================================
# 5. HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Lung Cancer Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive patient feature analysis and machine learning predictions'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="disclaimer">'
    '⚠️ Educational prototype only. Predictions are not medical '
    'diagnoses and must not be used to make treatment decisions.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================================
# 6. DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.subheader("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Input Features",
            len(feature_columns)
        )

    with col2:
        st.metric(
            "Numerical Features",
            len(numeric_columns)
        )

    with col3:
        st.metric(
            "Categorical Features",
            len(categorical_columns)
        )

    st.divider()

    st.subheader("About This Project")

    st.write("""
    This application uses a trained Random Forest classification
    pipeline to demonstrate predictions based on patient-related
    input features.

    The application accepts input values, applies the preprocessing
    steps saved with the model, and returns a predicted class.
    Where supported, it also displays the model's estimated
    probabilities for each class.
    """)

    left, right = st.columns(2)

    with left:
        st.markdown("### What you can do")

        st.write("""
        - Enter feature values interactively.
        - Generate a model prediction.
        - Examine estimated class probabilities.
        - Review the values supplied to the model.
        - Download a prediction summary.
        """)

    with right:
        st.markdown("### Machine learning workflow")

        st.write("""
        1. Data collection
        2. Data cleaning and EDA
        3. Data preprocessing
        4. Model training
        5. Model evaluation
        6. Application deployment
        """)

    st.info(
        "Select 'Patient Prediction' from the sidebar "
        "to enter data and generate a prediction."
    )

# =========================================================
# 7. PATIENT PREDICTION PAGE
# =========================================================

elif page == "Patient Prediction":

    st.subheader("🧑‍⚕️ Patient Feature Input")

    st.write(
        "Enter values for the features used during model training. "
        "Check every value before submitting."
    )

    with st.form("patient_form"):

        st.markdown("### Patient Information")

        values = {}

        numeric_features = [
            col for col in feature_columns
            if col in numeric_columns
        ]

        categorical_features = [
            col for col in feature_columns
            if col in categorical_columns
        ]

        # ---------------------------------------------
        # NUMERICAL INPUTS
        # ---------------------------------------------

        if numeric_features:

            st.markdown("#### Numerical Features")

            numeric_layout = st.columns(2)

            for i, col in enumerate(numeric_features):

                default = numeric_defaults.get(col, 0)

                with numeric_layout[i % 2]:

                    values[col] = st.number_input(
                        label=col.replace("_", " ").title(),
                        value=float(default),
                        key=f"num_{col}",
                        help=f"Enter the value for {col}."
                    )

        # ---------------------------------------------
        # CATEGORICAL INPUTS
        # ---------------------------------------------

        if categorical_features:

            st.markdown("#### Categorical Features")

            categorical_layout = st.columns(2)

            for i, col in enumerate(categorical_features):

                options = category_values.get(col, [])

                with categorical_layout[i % 2]:

                    if options:

                        values[col] = st.selectbox(
                            label=col.replace("_", " ").title(),
                            options=options,
                            key=f"cat_{col}",
                            help=f"Select a value for {col}."
                        )

                    else:

                        values[col] = st.text_input(
                            label=col.replace("_", " ").title(),
                            key=f"text_{col}"
                        )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Generate Prediction",
            use_container_width=True,
            type="primary"
        )

    # ---------------------------------------------
    # PREDICTION AND RESULTS
    # ---------------------------------------------

    if submitted:

        try:
            # Preserve the exact feature order from training
            new_patient = pd.DataFrame(
                [values],
                columns=feature_columns
            )

            # Generate the prediction
            prediction = model.predict(new_patient)[0]

            
            if prediction == 0:
                st.write("The patient will probably not survive.")

            else:
                st.write("The patient will survive.")    

            st.session_state["prediction"] = prediction
            st.session_state["patient_input"] = new_patient.copy()

            st.divider()

            st.subheader("📋 Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.markdown("#### Predicted Class")

                st.metric(
                    "Model Output",
                    str(prediction)
                )

                st.caption(
                    "Verify the target encoding before interpreting "
                    "this class as a survival outcome."
                )

            with result_col2:

                st.markdown("#### Model")

                st.metric(
                    "Algorithm",
                    "Random Forest"
                )

                st.caption(
                    "The saved pipeline includes the model "
                    "and its preprocessing steps."
                )

            # -----------------------------------------
            # CLASS PROBABILITIES
            # -----------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(new_patient)[0]
                classes = model.classes_

                probability_df = pd.DataFrame({
                    "Class": [str(cls) for cls in classes],
                    "Estimated Probability (%)": [
                        float(prob * 100)
                        for prob in probabilities
                    ]
                })

                st.divider()

                st.subheader("📈 Estimated Class Probabilities")

                st.caption(
                    "These probabilities are model estimates, "
                    "not validated clinical survival probabilities."
                )

                st.bar_chart(
                    probability_df.set_index("Class"),
                    y="Estimated Probability (%)"
                )

                probability_cols = st.columns(len(classes))

                for i, (cls, prob) in enumerate(
                    zip(classes, probabilities)
                ):

                    with probability_cols[i]:

                        st.metric(
                            f"Class {cls}",
                            f"{prob * 100:.2f}%"
                        )

            # -----------------------------------------
            # INPUT SUMMARY
            # -----------------------------------------

            st.divider()

            with st.expander("Review entered feature values"):

                st.dataframe(
                    new_patient.T.rename(
                        columns={0: "Entered Value"}
                    ),
                    use_container_width=True
                )

            # -----------------------------------------
            # DOWNLOAD RESULTS
            # -----------------------------------------

            output = new_patient.copy()
            output["predicted_class"] = str(prediction)

            if hasattr(model, "predict_proba"):

                for cls, prob in zip(classes, probabilities):

                    output[
                        f"estimated_probability_class_{cls}"
                    ] = float(prob)

            csv_data = output.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction as CSV",
                data=csv_data,
                file_name="lung_cancer_prediction.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success("Prediction generated successfully.")

        except Exception as error:

            st.error(
                "The prediction could not be generated. "
                "Check the feature values, data types, and model schema."
            )

            st.exception(error)

# =========================================================
# 8. MODEL INFORMATION PAGE
# =========================================================

elif page == "Model Information":

    st.subheader("🤖 Model Information")

    st.write("### Algorithm")

    st.write(
        "Random Forest Classifier — an ensemble learning algorithm "
        "that combines predictions from multiple decision trees."
    )

    st.write("### Input Schema")

    schema_df = pd.DataFrame({
        "Feature": feature_columns,
        "Feature Type": [
            "Numerical" if col in numeric_columns
            else "Categorical" if col in categorical_columns
            else "Other"
            for col in feature_columns
        ]
    })

    st.dataframe(
        schema_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("### Model Classes")

    try:
        st.write([
            str(cls) for cls in model.classes_
        ])

    except AttributeError:
        st.write(
            "The class labels could not be retrieved "
            "from the saved pipeline."
        )

    st.write("### Limitations")

    st.markdown("""
    - The model can only process features compatible with its training schema.
    - Predictions depend on the quality and representativeness of the training data.
    - Estimated probabilities may not be well calibrated.
    - Performance must be evaluated on held-out data.
    - The application has not been established as a clinically validated tool.
    - Patient information should not be entered into a public demonstration app.
    """)

    st.info(
        "To assess the model, report test-set precision, recall, "
        "F1-score, confusion matrix, and other suitable metrics."
    )

# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Lung Cancer Analysis | Machine Learning Project | "
    "Educational Prototype"
)
