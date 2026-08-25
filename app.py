import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from groq import Groq

# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI-Powered Loan Approval Assistant",
    page_icon="💰",
    layout="wide"
)

# ============================================================
# 2. GROQ SETUP
# ============================================================

groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    groq_client = Groq(api_key=groq_key)
else:
    groq_client = None


# ============================================================
# 3. LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_ml_objects():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        base_dir, "models", "loan_model.joblib"
    )

    preprocessor_path = os.path.join(
        base_dir, "models", "preprocessor.joblib"
    )

    feature_names_path = os.path.join(
        base_dir, "models", "feature_names.joblib"
    )

    if not (
        os.path.exists(model_path)
        and os.path.exists(preprocessor_path)
        and os.path.exists(feature_names_path)
    ):
        return None, None, None

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_names = joblib.load(feature_names_path)

    return model, preprocessor, feature_names


model, preprocessor, feature_names = load_ml_objects()


st.title("💰 AI-Powered Loan Approval Assistant")
st.markdown("---")

# ============================================================
# 5. CHECK MODEL
# ============================================================

if model is None or preprocessor is None:

    st.error(
        """
        ❌ Model files could not be loaded.

        Please make sure these files exist inside the `models` folder:

        - `loan_model.joblib`
        - `preprocessor.joblib`
        - `feature_names.joblib`
        """
    )

    st.stop()


# ============================================================
# 6. LOAN APPLICATION INPUT
# ============================================================

st.header("📋 Loan Application Details")

col1, col2, col3 = st.columns(3)


# ---------------- PERSONAL DETAILS ----------------

with col1:

    st.subheader("👤 Personal Details")

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

    education = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self-Employed?",
        ["No", "Yes"]
    )


# ---------------- LOAN DETAILS ----------------

with col2:

    st.subheader("💵 Loan & Credit Information")

    cibil_score = st.slider(
        "CIBIL (Credit) Score",
        min_value=300,
        max_value=900,
        value=650,
        step=1
    )

    income_annum = st.number_input(
        "Annual Income ($)",
        min_value=10000,
        max_value=10000000,
        value=2000000,
        step=50000,
        format="%d"
    )

    loan_amount = st.number_input(
        "Requested Loan Amount ($)",
        min_value=10000,
        max_value=50000000,
        value=5000000,
        step=100000,
        format="%d"
    )

    loan_term = st.slider(
        "Loan Term (Years)",
        min_value=2,
        max_value=20,
        value=10,
        step=2
    )


# ---------------- ASSETS ----------------

with col3:

    st.subheader("🏡 Asset Portfolio")

    residential_assets_value = st.number_input(
        "Residential Asset Value ($)",
        min_value=0,
        max_value=50000000,
        value=1500000,
        step=50000,
        format="%d"
    )

    commercial_assets_value = st.number_input(
        "Commercial Asset Value ($)",
        min_value=0,
        max_value=50000000,
        value=1000000,
        step=50000,
        format="%d"
    )

    luxury_assets_value = st.number_input(
        "Luxury Asset Value ($)",
        min_value=0,
        max_value=50000000,
        value=2000000,
        step=50000,
        format="%d"
    )

    bank_asset_value = st.number_input(
        "Bank Balance Asset Value ($)",
        min_value=0,
        max_value=50000000,
        value=500000,
        step=100000,
        format="%d"
    )


# ============================================================
# 7. PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_btn = st.button(
    "🚀 Predict Loan Approval",
    use_container_width=True
)


# ============================================================
# 8. PREDICTION
# ============================================================

if predict_btn:

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "no_of_dependents": no_of_dependents,

        "education": education,

        "self_employed": self_employed,

        "income_annum": income_annum,

        "loan_amount": loan_amount,

        "loan_term": loan_term,

        "cibil_score": cibil_score,

        "residential_assets_value": residential_assets_value,

        "commercial_assets_value": commercial_assets_value,

        "luxury_assets_value": luxury_assets_value,

        "bank_asset_value": bank_asset_value

    }])


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    try:

        input_processed = preprocessor.transform(input_data)

    except Exception as e:

        st.error(f"❌ Preprocessing error: {e}")

        st.stop()


    # --------------------------------------------------------
    # ML Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_processed)[0]

    probabilities = model.predict_proba(input_processed)[0]

    rejection_prob = probabilities[0] * 100
    approval_prob = probabilities[1] * 100


    # ========================================================
    # 9. DISPLAY ML RESULT
    # ========================================================

    st.markdown("---")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.header("📊 ML Prediction")

        if prediction == 1:

            st.success("🎉 LOAN APPROVED")

            st.metric(
                "Approval Probability",
                f"{approval_prob:.1f}%"
            )

        else:

            st.error("❌ LOAN REJECTED")

            st.metric(
                "Rejection Probability",
                f"{rejection_prob:.1f}%"
            )

        st.info(
            """
            ℹ️ This is an educational ML prediction system.
            It is not an actual banking approval decision.
            """
        )


    # ========================================================
    # 10. CALCULATE SUPPORTING INFORMATION
    # ========================================================

    total_assets = (
        residential_assets_value
        + commercial_assets_value
        + luxury_assets_value
        + bank_asset_value
    )

    annual_payment_estimation = loan_amount / loan_term

    income_ratio = (
        annual_payment_estimation / income_annum
    ) * 100


    # ========================================================
    # 11. GROQ AI EXPLANATION
    # ========================================================

    with result_col2:

        st.header("🤖 AI Explanation")

        if groq_client is None:

            st.warning(
                """
                ⚠️ Groq API key was not found.

                Please check your `.env` file.
                """
            )

        else:

            with st.spinner("🤖 Groq AI is analyzing the prediction..."):

                try:

                    result_text = (
                        "LOAN APPROVED"
                        if prediction == 1
                        else "LOAN REJECTED"
                    )

                    prompt = f"""
You are an AI assistant explaining the result of an educational
machine-learning loan prediction system.

IMPORTANT:
- The Random Forest model has already made the prediction.
- Do NOT change or override the prediction.
- Do NOT claim that the prediction is a real banking decision.
- Explain the result clearly and simply.
- Mention important positive and negative factors.
- Do not provide personalized financial advice.
- Keep the explanation between 4 and 6 bullet points.

ML Prediction:
{result_text}

Approval Probability:
{approval_prob:.1f}%

Rejection Probability:
{rejection_prob:.1f}%

Applicant Information:

Number of Dependents:
{no_of_dependents}

Education:
{education}

Self Employed:
{self_employed}

CIBIL Score:
{cibil_score}

Annual Income:
${income_annum:,.0f}

Requested Loan Amount:
${loan_amount:,.0f}

Loan Term:
{loan_term} years

Total Assets:
${total_assets:,.0f}

Estimated Annual Loan Amount:
${annual_payment_estimation:,.2f}

Estimated Loan Amount / Annual Income:
{income_ratio:.1f}%

Explain why the machine-learning model may have produced this
prediction based on the provided information.
"""

                    chat_completion = groq_client.chat.completions.create(

                        model="openai/gpt-oss-120b",

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a clear and responsible "
                                    "AI explanation assistant."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        temperature=0.3,

                        max_tokens=500
                    )

                    ai_response = (
                        chat_completion
                        .choices[0]
                        .message
                        .content
                    )

                    st.markdown(ai_response)

                except Exception as e:

                    st.error(
                        f"❌ Groq AI Error: {e}"
                    )


    # ========================================================
    # 12. APPLICATION SUMMARY
    # ========================================================

    st.markdown("---")

    st.header("📋 Application Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "CIBIL Score",
            cibil_score
        )

        st.metric(
            "Annual Income",
            f"${income_annum:,.0f}"
        )

    with summary_col2:

        st.metric(
            "Loan Amount",
            f"${loan_amount:,.0f}"
        )

        st.metric(
            "Loan Term",
            f"{loan_term} years"
        )

    with summary_col3:

        st.metric(
            "Total Assets",
            f"${total_assets:,.0f}"
        )

        st.metric(
            "Income-to-Loan Estimate",
            f"{income_ratio:.1f}%"
        )


    # ========================================================
    # 13. FEATURE IMPORTANCE
    # ========================================================

    st.markdown("---")

    st.subheader("📊 Random Forest Feature Importance")

    st.write(
        """
        This chart shows which features the trained Random Forest
        model considers important across the training data.
        """
    )

    try:

        importances = model.feature_importances_

        importance_df = pd.DataFrame({

            "Feature": [
                name.replace("_", " ").title()
                for name in feature_names
            ],

            "Importance": importances

        }).sort_values(
            by="Importance",
            ascending=True
        )

        st.bar_chart(
            importance_df.set_index("Feature")["Importance"],
            horizontal=True,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Feature importance chart could not be displayed: {e}"
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🤖 AI-Powered Loan Approval Assistant | "
    "Random Forest + Groq Generative AI"
)